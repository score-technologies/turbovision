import os, time, socket, asyncio, logging, signal

from aiohttp import web
import bittensor as bt

from scorevision.utils.bittensor_helpers import get_last_update_for_hotkey
from scorevision.utils.settings import get_settings

logger = logging.getLogger("sv-signer")

NETUID = int(os.getenv("SCOREVISION_NETUID", "44"))
MECHID = 1

# Global shutdown event
shutdown_event = asyncio.Event()


async def get_subtensor():
    settings = get_settings()
    ep = settings.BITTENSOR_SUBTENSOR_ENDPOINT
    fb = settings.BITTENSOR_SUBTENSOR_FALLBACK
    st = bt.async_subtensor(ep)
    try:
        await st.initialize()
        return st
    except Exception:
        logger.warning("Subtensor init failed; fallback to %s", fb)
        st = bt.async_subtensor(fb)
        await st.initialize()
        return st


async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    mechid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "[signer]",
) -> bool:
    """"""
    settings = get_settings()
    confirm_blocks = max(1, int(os.getenv("SIGNER_CONFIRM_BLOCKS", "3")))
    for attempt in range(retries):
        st = None
        sync_st = None
        try:
            st = await get_subtensor()
            ref_block = await st.get_current_block()

            # extrinsic - use sync subtensor for set_weights call
            sync_st = bt.subtensor(settings.BITTENSOR_SUBTENSOR_ENDPOINT)
            success, message = sync_st.set_weights(
                wallet=wallet,
                netuid=netuid,
                mechid=mechid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=wait_for_inclusion,
            )

            if not success:
                logger.warning(
                    "%s extrinsic submit failed: %s",
                    log_prefix,
                    message or "unknown error",
                )
            else:
                logger.info(
                    "%s extrinsic submitted; monitoring up to %d blocks … (ref=%d, msg=%s)",
                    log_prefix,
                    confirm_blocks,
                    ref_block,
                    message or "",
                )

                latest_lu = None
                hotkey = wallet.hotkey.ss58_address
                for wait_idx in range(confirm_blocks):
                    await st.wait_for_block()
                    meta = await st.metagraph(netuid, mechid=mechid)
                    try:
                        meta_hotkeys = getattr(meta, "hotkeys", []) or []
                        try:
                            hotkey_present = hotkey in meta_hotkeys
                        except TypeError:
                            try:
                                hotkey_present = hotkey in list(meta_hotkeys)
                            except TypeError:
                                hotkey_present = False
                        
                        if not hotkey_present:
                            logger.warning(
                                "%s wallet hotkey not found in metagraph; retry…",
                                log_prefix,
                            )
                            break

                        latest_lu = get_last_update_for_hotkey(
                            meta, hotkey, pubkey_hex=wallet.hotkey.public_key.hex()
                        )
                        if latest_lu is None:
                            logger.warning(
                                "%s wallet hotkey found but no last_update entry; retry…",
                                log_prefix,
                            )
                            break
                        if latest_lu >= ref_block:
                            logger.info(
                                "%s confirmation OK (last_update=%d >= ref=%d after %d block(s))",
                                log_prefix,
                                latest_lu,
                                ref_block,
                                wait_idx + 1,
                            )
                            return True
                        logger.debug(
                            "%s waiting for inclusion… (last_update=%d < ref=%d, waited %d/%d block(s))",
                            log_prefix,
                            latest_lu,
                            ref_block,
                            wait_idx + 1,
                            confirm_blocks,
                        )
                    finally:
                        # Clean up metagraph object to prevent memory accumulation
                        del meta

                if latest_lu is not None:
                    logger.warning(
                        "%s not yet included after %d blocks (last_update=%d < ref=%d), retry…",
                        log_prefix,
                        confirm_blocks,
                        latest_lu,
                        ref_block,
                    )
                else:
                    logger.warning(
                        "%s not yet included after %d blocks (hotkey missing), retry…",
                        log_prefix,
                        confirm_blocks,
                    )
        except Exception as e:
            logger.warning(
                "%s attempt %d error: %s: %s",
                log_prefix,
                attempt + 1,
                type(e).__name__,
                e,
            )
        finally:
            # Clean up connections to prevent memory leaks
            if st is not None:
                try:
                    await st.close()
                except Exception:
                    pass
            if sync_st is not None:
                try:
                    sync_st.close()
                except Exception:
                    pass
        await asyncio.sleep(delay_s)
    return False


async def run_signer() -> None:
    settings = get_settings()
    host = settings.SIGNER_HOST
    port = settings.SIGNER_PORT

    # Set up signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal, stopping signer...")
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    # Wallet Bittensor
    cold = settings.BITTENSOR_WALLET_COLD
    hot = settings.BITTENSOR_WALLET_HOT
    wallet = bt.wallet(name=cold, hotkey=hot)

    @web.middleware
    async def access_log(request: web.Request, handler):
        t0 = time.monotonic()
        try:
            resp = await handler(request)
            return resp
        finally:
            dt = (time.monotonic() - t0) * 1000
            logger.info(
                "[signer] %s %s -> %s %.1fms",
                request.method,
                request.path,
                getattr(getattr(request, "response", None), "status", "?"),
                dt,
            )

    async def health(_req: web.Request):
        return web.json_response({"ok": True})

    async def sign_handler(req: web.Request):
        """"""
        try:
            payload = await req.json()
            data = payload.get("payloads") or payload.get("data") or []
            if isinstance(data, str):
                data = [data]
            sigs = [(wallet.hotkey.sign(data=d.encode("utf-8"))).hex() for d in data]
            return web.json_response(
                {
                    "success": True,
                    "signatures": sigs,
                    "hotkey": wallet.hotkey.ss58_address,
                }
            )
        except Exception as e:
            logger.error("[sign] error: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def set_weights_handler(req: web.Request):
        """"""
        try:
            payload = await req.json()
            netuid = int(payload.get("netuid", NETUID))
            default_mechid = getattr(settings, "SCOREVISION_MECHID", MECHID)
            mechid = int(payload.get("mechid", default_mechid))
            uids = payload.get("uids") or []
            wgts = payload.get("weights") or []
            wfi = bool(payload.get("wait_for_inclusion", False))

            if isinstance(uids, int):
                uids = [uids]
            if isinstance(wgts, (int, float, str)):
                try:
                    wgts = [float(wgts)]
                except:
                    wgts = [0.0]
            if not isinstance(uids, list):
                uids = list(uids)
            if not isinstance(wgts, list):
                wgts = list(wgts)
            try:
                uids = [int(u) for u in uids]
            except:
                uids = []
            try:
                wgts = [float(w) for w in wgts]
            except:
                wgts = []

            if len(uids) != len(wgts) or not uids:
                return web.json_response(
                    {"success": False, "error": "uids/weights mismatch or empty"},
                    status=400,
                )

            ok = await _set_weights_with_confirmation(
                wallet,
                netuid,
                mechid,
                uids,
                wgts,
                wfi,
                retries=int(os.getenv("SIGNER_RETRIES", "10")),
                delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                log_prefix="[signer]",
            )
            return web.json_response(
                (
                    {"success": True}
                    if ok
                    else {"success": False, "error": "confirmation failed"}
                ),
                status=200 if ok else 500,
            )
        except Exception as e:
            logger.error("[set_weights] error: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    app = web.Application(middlewares=[access_log])
    app.add_routes(
        [
            web.get("/healthz", health),
            web.post("/sign", sign_handler),
            web.post("/set_weights", set_weights_handler),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    try:
        hn = socket.gethostname()
        ip = socket.gethostbyname(hn)
    except Exception:
        hn, ip = ("?", "?")
    logger.info(
        "Signer listening on http://%s:%s hostname=%s ip=%s", host, port, hn, ip
    )

    # Wait for shutdown signal instead of infinite loop
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down signer...")
        await runner.cleanup()
