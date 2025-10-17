import os, time, socket, asyncio, logging, signal, gc, threading
from typing import Tuple

from aiohttp import web
import bittensor as bt

from scorevision.utils.bittensor_helpers import get_last_update_for_hotkey
from scorevision.utils.settings import get_settings

logger = logging.getLogger("sv-signer")

NETUID = int(os.getenv("SCOREVISION_NETUID", "44"))
MECHID = 1

# Global shutdown event
shutdown_event = asyncio.Event()

_ASYNC_SUBTENSOR: bt.AsyncSubtensor | None = None
_ASYNC_SUBTENSOR_LOCK = asyncio.Lock()
_SYNC_SUBTENSOR: bt.Subtensor | None = None
_SYNC_SUBTENSOR_LOCK = threading.Lock()


async def get_subtensor():
    global _ASYNC_SUBTENSOR
    async with _ASYNC_SUBTENSOR_LOCK:
        if _ASYNC_SUBTENSOR is not None:
            return _ASYNC_SUBTENSOR
        settings = get_settings()
        ep = settings.BITTENSOR_SUBTENSOR_ENDPOINT
        fb = settings.BITTENSOR_SUBTENSOR_FALLBACK
        for endpoint in (ep, fb):
            try:
                st = bt.async_subtensor(endpoint)
                await st.initialize()
                _ASYNC_SUBTENSOR = st
                if endpoint != ep:
                    logger.warning("Subtensor init fell back to %s", endpoint)
                break
            except Exception as e:
                logger.warning("Subtensor init failed for %s: %s", endpoint, e)
                continue
        if _ASYNC_SUBTENSOR is None:
            raise RuntimeError("Unable to initialize async subtensor")
        return _ASYNC_SUBTENSOR


def _get_sync_subtensor() -> bt.Subtensor:
    global _SYNC_SUBTENSOR
    with _SYNC_SUBTENSOR_LOCK:
        if _SYNC_SUBTENSOR is not None:
            return _SYNC_SUBTENSOR
        settings = get_settings()
        ep = settings.BITTENSOR_SUBTENSOR_ENDPOINT
        fb = settings.BITTENSOR_SUBTENSOR_FALLBACK
        for endpoint in (ep, fb):
            try:
                st = bt.subtensor(endpoint)
                _SYNC_SUBTENSOR = st
                if endpoint != ep:
                    logger.warning("Sync subtensor init fell back to %s", endpoint)
                break
            except Exception as e:
                logger.warning("Sync subtensor init failed for %s: %s", endpoint, e)
                continue
        if _SYNC_SUBTENSOR is None:
            raise RuntimeError("Unable to initialize sync subtensor")
        return _SYNC_SUBTENSOR


async def _reset_async_subtensor():
    global _ASYNC_SUBTENSOR
    async with _ASYNC_SUBTENSOR_LOCK:
        if _ASYNC_SUBTENSOR is not None:
            try:
                await _ASYNC_SUBTENSOR.close()
            except Exception:
                pass
            _ASYNC_SUBTENSOR = None


def _reset_sync_subtensor():
    global _SYNC_SUBTENSOR
    with _SYNC_SUBTENSOR_LOCK:
        if _SYNC_SUBTENSOR is not None:
            try:
                _SYNC_SUBTENSOR.close()
            except Exception:
                pass
            _SYNC_SUBTENSOR = None


async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    mechid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool,
    wait_for_finalization: bool,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "[signer]",
) -> bool:
    """ """
    settings = get_settings()
    confirm_blocks = max(0, int(os.getenv("SIGNER_CONFIRM_BLOCKS", "6")))
    earliest_ref_block = None
    latest_known_update = None
    baseline_last_update = None
    for attempt in range(retries):
        st = None
        sync_st = None
        try:
            st = await get_subtensor()

            if baseline_last_update is None:
                meta_snapshot = await st.metagraph(netuid, mechid=mechid)
                try:
                    hotkey = wallet.hotkey.ss58_address
                    hotkeys = getattr(meta_snapshot, "hotkeys", []) or []
                    try:
                        present = hotkey in hotkeys
                    except TypeError:
                        try:
                            present = hotkey in list(hotkeys)
                        except TypeError:
                            present = False
                    if present:
                        baseline_last_update = get_last_update_for_hotkey(
                            meta_snapshot,
                            hotkey,
                            pubkey_hex=wallet.hotkey.public_key.hex(),
                        )
                        latest_known_update = baseline_last_update
                finally:
                    del meta_snapshot

            ref_block = await st.get_current_block()
            if earliest_ref_block is None:
                earliest_ref_block = ref_block

            if latest_known_update is not None and earliest_ref_block is not None:
                if latest_known_update >= earliest_ref_block:
                    logger.info(
                        "%s existing confirmation detected (last_update=%d >= ref=%d); skipping resend.",
                        log_prefix,
                        latest_known_update,
                        earliest_ref_block,
                    )
                    return True

            try:
                sync_st = _get_sync_subtensor()
            except Exception:
                _reset_sync_subtensor()
                sync_st = _get_sync_subtensor()

            try:
                success, message = sync_st.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    mechid=mechid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                msg_str = str(message or "")
            except Exception as e:
                msg_str = f"{type(e).__name__}: {e}"
                if "SettingWeightsTooFast" in msg_str:
                    logger.error(
                        "%s SettingWeightsTooFast (exception) → weights are likely set working on confirmation",
                        log_prefix,
                    )
                    return True
                raise

            if not success:
                if "SettingWeightsTooFast" in msg_str:
                    logger.error(
                        "%s SettingWeightsTooFast (return) → weights are likely set working on confirmation",
                        log_prefix,
                    )
                    return True
                logger.warning("%s extrinsic submit failed: %s", log_prefix, msg_str)

            else:
                if confirm_blocks <= 0:
                    logger.info(
                        "%s extrinsic submitted; confirmation skipped (confirm_blocks=0) → weights are likely set working on confirmation",
                        log_prefix,
                    )
                    return True

                logger.info(
                    "%s extrinsic submitted; monitoring up to %d blocks … (ref=%d, msg=%s)",
                    log_prefix,
                    confirm_blocks,
                    ref_block,
                    msg_str,
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
                                "%s wallet hotkey not found in metagraph; continue waiting…",
                                log_prefix,
                            )
                            continue

                        latest_lu = get_last_update_for_hotkey(
                            meta, hotkey, pubkey_hex=wallet.hotkey.public_key.hex()
                        )
                        if latest_lu is None:
                            logger.warning(
                                "%s wallet hotkey found but no last_update entry; continue waiting…",
                                log_prefix,
                            )
                            continue
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
                        del meta
                        if latest_lu is not None:
                            latest_known_update = max(
                                latest_known_update or -1, latest_lu
                            )

                logger.warning(
                    "%s not yet included after %d blocks (last_update=%s < ref=%d) → weights are likely set working on confirmation",
                    log_prefix,
                    confirm_blocks,
                    "None" if latest_lu is None else latest_lu,
                    ref_block,
                )
                return True

        except Exception as e:
            logger.warning(
                "%s attempt %d error: %s: %s",
                log_prefix,
                attempt + 1,
                type(e).__name__,
                e,
            )
            await _reset_async_subtensor()
            _reset_sync_subtensor()
        finally:
            gc.collect()
        await asyncio.sleep(delay_s)
    return False


async def run_signer() -> None:
    settings = get_settings()
    host = settings.SIGNER_HOST
    port = settings.SIGNER_PORT

    def signal_handler():
        logger.info("Received shutdown signal, stopping signer...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

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
            wff = bool(payload.get("wait_for_finalization", False))

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
                wff,
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
        finally:
            gc.collect()

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
        await _reset_async_subtensor()
        _reset_sync_subtensor()
        gc.collect()
