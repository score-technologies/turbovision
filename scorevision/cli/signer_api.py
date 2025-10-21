import os, time, socket, asyncio, logging, signal, gc, threading
from typing import Tuple

from aiohttp import web
import bittensor as bt

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


def _set_weights(
    *,
    wallet: "bt.wallet",
    netuid: int,
    mechid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool,
    wait_for_finalization: bool,
    log_prefix: str = "[signer]",
) -> bool:
    """ """
    try:
        st = _get_sync_subtensor()
    except Exception:
        _reset_sync_subtensor()
        st = _get_sync_subtensor()

    try:
        success, message = st.set_weights(
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
                f"{log_prefix} SettingWeightsTooFast (exception) → treating as success."
            )
            return True
        logger.warning(f"{log_prefix} set_weights exception: {msg_str}")
        return False

    if success:
        logger.info(f"{log_prefix} set_weights success.")
        return True

    if "SettingWeightsTooFast" in msg_str:
        logger.error(
            f"{log_prefix} SettingWeightsTooFast (return) → treating as success."
        )
        return True

    logger.warning(f"{log_prefix} set_weights failed: {msg_str or 'unknown error'}")
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
                except Exception:
                    wgts = [0.0]
            if not isinstance(uids, list):
                uids = list(uids)
            if not isinstance(wgts, list):
                wgts = list(wgts)
            try:
                uids = [int(u) for u in uids]
            except Exception:
                uids = []
            try:
                wgts = [float(w) for w in wgts]
            except Exception:
                wgts = []

            if len(uids) != len(wgts) or not uids:
                return web.json_response(
                    {"success": False, "error": "uids/weights mismatch or empty"},
                    status=400,
                )

            ok = _set_weights(
                wallet=wallet,
                netuid=netuid,
                mechid=mechid,
                uids=uids,
                weights=wgts,
                wait_for_inclusion=wfi,
                wait_for_finalization=wff,
                log_prefix="[signer]",
            )
            return web.json_response(
                (
                    {"success": True}
                    if ok
                    else {"success": False, "error": "set_weights failed"}
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
