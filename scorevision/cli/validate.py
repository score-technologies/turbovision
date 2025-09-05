import os
import time
import asyncio
import logging
import traceback

import aiohttp
import bittensor as bt

# ---- Imports internes ScoreVision ----
from scorevision.utils.cloudflare_helpers import dataset_sv
from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    _set_weights_with_confirmation,
)
from scorevision.utils.prometheus import (
    LASTSET_GAUGE,
    CACHE_DIR,
    CACHE_FILES,
    EMA_BY_UID,
    CURRENT_WINNER,
)
from scorevision.utils.settings import get_settings

logger = logging.getLogger("scorevision.validator")

for noisy in ["websockets", "websockets.client", "substrateinterface", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def _validate_main(tail: int, alpha: float, m_min: int, tempo: int):

    settings = get_settings()

    NETUID = settings.SCOREVISION_NETUID

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    st = None
    last_done = -1
    while True:
        try:
            if st is None:
                st = await get_subtensor()
            block = await st.get_current_block()

            if block % tempo != 0 or block <= last_done:
                await st.wait_for_block()
                continue

            uids, weights = await get_weights(tail=tail, alpha=alpha, m_min=m_min)
            if not uids:
                logger.warning("No eligible uids this round; skipping.")
                last_done = block
                continue

            ok = await retry_set_weights(wallet, uids, weights)
            if ok:
                LASTSET_GAUGE.set(time.time())
                logger.info("set_weights OK at block %d", block)
            else:
                logger.warning("set_weights failed at block %d", block)

            try:
                sz = sum(
                    f.stat().st_size for f in CACHE_DIR.glob("*.jsonl") if f.is_file()
                )
                CACHE_FILES.set(len(list(CACHE_DIR.glob("*.jsonl"))))
            except Exception:
                pass

            last_done = block

        except asyncio.CancelledError:
            break
        except Exception as e:
            traceback.print_exc()
            logger.warning("Validator loop error: %s — reconnecting…", e)
            st = None
            await asyncio.sleep(5)


# ---------------- Weights selection ---------------- #
async def get_weights(tail: int = 28800, alpha: float = 0.2, m_min: int = 25):
    """ """
    settings = get_settings()
    st = await get_subtensor()
    NETUID = settings.SCOREVISION_NETUID
    meta = await st.metagraph(NETUID)
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys)}

    ema: dict[str, float] = {}
    cnt: dict[str, int] = {}

    async for line in dataset_sv(tail):
        try:
            payload = line.get("payload") or {}
            miner = payload.get("miner") or {}
            hk = (miner.get("hotkey") or "").strip()
            if not hk or hk not in hk_to_uid:
                continue
            score = float(((payload.get("evaluation") or {}).get("score")) or 0.0)
        except Exception:
            continue

        if hk not in ema:
            ema[hk] = score
            cnt[hk] = 1
        else:
            ema[hk] = alpha * score + (1 - alpha) * ema[hk]
            cnt[hk] += 1

    if not ema:
        logger.warning("No data → default weight to uid 0")
        return [0], [1.0]

    elig_hk = [hk for hk, n in cnt.items() if n >= m_min] or list(ema.keys())
    elig_hk = [hk for hk in elig_hk if hk in hk_to_uid]
    winner_hk = max(elig_hk, key=lambda k: ema.get(k, 0.0))
    winner_uid = hk_to_uid.get(winner_hk, 0)

    # Poids clairsemés (plus léger on-chain)
    uids = [winner_uid]
    weights = [1.0]

    # Prometheus (optionnel)
    for hk, v in ema.items():
        uid = hk_to_uid.get(hk)
        if uid is not None:
            EMA_BY_UID.labels(uid=str(uid)).set(v)
    CURRENT_WINNER.set(winner_uid)

    logger.info(
        "Winner hk=%s uid=%d EMA=%.4f (n=%d)",
        winner_hk[:8] + "…",
        winner_uid,
        ema[winner_hk],
        cnt[winner_hk],
    )
    return uids, weights


async def retry_set_weights(wallet, uids, weights):
    """
    1) Tente /set_weights du signer (HTTP)
    2) Fallback: set_weights local + confirmation par lecture du metagraph
    """
    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID
    signer_url = settings.SIGNER_URL

    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "uids": uids,
                    "weights": weights,
                    "wait_for_inclusion": False,
                },
            )
            try:
                data = await resp.json()
            except Exception:
                data = {"raw": await resp.text()}
            if resp.status == 200 and data.get("success"):
                return True
            logger.warning("Signer error status=%s body=%s", resp.status, data)
    except aiohttp.ClientConnectorError as e:
        logger.info("Signer unreachable: %s — falling back to local set_weights", e)
    except asyncio.TimeoutError:
        logger.warning("Signer timed out — falling back to local set_weights")

    # ---- Fallback local ----
    return await _set_weights_with_confirmation(wallet, NETUID, uids, weights)
