import asyncio
import logging
import os
import signal
import time
import traceback
import gc
from functools import lru_cache
import aiohttp

import bittensor as bt
from scorevision.utils.settings import get_settings
from scorevision.utils.windows import get_current_window_id
from scorevision.utils.prometheus import (
    LASTSET_GAUGE,
    CACHE_DIR,
    CACHE_FILES,
    EMA_BY_UID,
    CURRENT_WINNER,
    VALIDATOR_BLOCK_HEIGHT,
    VALIDATOR_LOOP_TOTAL,
    VALIDATOR_LAST_BLOCK_SUCCESS,
    VALIDATOR_WEIGHT_FAIL_TOTAL,
    VALIDATOR_CACHE_BYTES,
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
    VALIDATOR_COMMIT_TOTAL,
    VALIDATOR_LAST_LOOP_DURATION_SECONDS,
    VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS,
)

from scorevision.utils.bittensor_helpers import reset_subtensor, get_subtensor
from scorevision.utils.window_scores import (
    aggregate_window_shards,
    compute_winner_from_window,
)
from scorevision.utils.ewma import (
    calculate_ewma_alpha,
    load_previous_ewma,
    save_ewma,
    update_ewma_score,
)

logger = logging.getLogger("scorevision.validator")
shutdown_event = asyncio.Event()


async def retry_set_weights(wallet, uids, weights):

    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID
    MECHID = settings.SCOREVISION_MECHID
    signer_url = settings.SIGNER_URL

    loop = asyncio.get_running_loop()
    request_start = loop.time()
    try:
        timeout = aiohttp.ClientTimeout(connect=2, total=300)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "mechid": MECHID,
                    "uids": uids,
                    "weights": weights,
                    "wait_for_inclusion": True,
                    "wait_for_finalization": True,
                },
            )
            try:
                data = await resp.json()
            except Exception:
                data = {"raw": await resp.text()}

            duration = loop.time() - request_start
            VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS.set(duration)

            if resp.status == 200 and data.get("success"):
                return True

            body_txt = ""
            try:
                body_txt = (
                    data
                    if isinstance(data, str)
                    else (data.get("error") or data.get("raw") or "")
                )
            except Exception:
                pass
            if "SettingWeightsTooFast" in str(body_txt):
                logger.warning(
                    "Signer returns SettingWeightsTooFast; weights are likely set working on confirmation."
                )
                return True

            VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_http").inc()
            logger.warning("Signer error status=%s body=%s", resp.status, data)

    except aiohttp.ClientConnectorError as e:
        VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS.set(loop.time() - request_start)
        logger.warning("Signer unreachable: %s — skipping local fallback", e)
        VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_connect").inc()
    except asyncio.TimeoutError:
        VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS.set(loop.time() - request_start)
        logger.warning(
            "Signer timed out — weights are likely set working on confirmation"
        )
        VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_timeout").inc()

    return False


@lru_cache(maxsize=1)
def _validator_hotkey_ss58() -> str:
    settings = get_settings()
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return wallet.hotkey.ss58_address


async def _validate_main(tail: int, alpha: float, m_min: int, tempo: int) -> None:
    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID

    def signal_handler():
        logger.info("Received shutdown signal, stopping validator...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    st = None
    last_done = -1
    while not shutdown_event.is_set():
        try:
            if st is None:
                st = await get_subtensor()
            block = await st.get_current_block()
            VALIDATOR_BLOCK_HEIGHT.set(block)
            current_window_id = get_current_window_id(block, tempo=tempo)
            logger.info(
                "[validator] current_window_id=%s (block=%d, tempo=%d)",
                current_window_id,
                block,
                tempo,
            )

            if block % tempo != 0 or block <= last_done:
                try:
                    await asyncio.wait_for(st.wait_for_block(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning(
                        "wait_for_block error (%s); resetting subtensor", err
                    )
                    VALIDATOR_LOOP_TOTAL.labels(outcome="subtensor_error").inc()
                    reset_subtensor()
                    st = None
                    await asyncio.sleep(2.0)
                    continue
                continue

            iter_loop = asyncio.get_running_loop()
            iter_start = iter_loop.time()
            loop_outcome = "unknown"

            try:
                # ------------------------------
                # NEW: Aggregate shards via window_scores.py
                # ------------------------------
                window_summary_file = await aggregate_window_shards(
                    cache_root=CACHE_DIR,
                    tail=tail,
                    window_id=current_window_id,
                    min_samples=m_min,
                )

                if not window_summary_file.exists():
                    logger.warning(
                        "No eligible miners in window; skipping block %d", block
                    )
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_uids").inc()
                    loop_outcome = "no_uids"
                    last_done = block
                    continue

                # ---------------------------------------------------------
                # EWMA INTEGRATION
                # ---------------------------------------------------------

                # Compute the per-window clip-mean winners
                uids, clip_means = await compute_winner_from_window(window_summary_file)

                if not uids:
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_uids").inc()
                    last_done = block
                    continue

                # EWMA alpha from settings (half-life in windows)
                alpha = calculate_ewma_alpha(settings.SCOREVISION_WINDOW_HALF_LIFE)

                # Previous window id: on recule d'un tempo en bloc
                prev_window_id = get_current_window_id(block - tempo, tempo=tempo)
                prev_scores = load_previous_ewma(prev_window_id)

                # Build current score dict
                current_scores = {uid: score for uid, score in zip(uids, clip_means)}

                # Compute EWMA: S_t = α×current + (1−α)×prev
                ewma_scores = {}
                for uid, score in current_scores.items():
                    prev = prev_scores.get(str(uid))
                    ewma_scores[str(uid)] = update_ewma_score(
                        current_score=score,
                        previous_ewma=prev,
                        alpha=alpha,
                    )

                # Persist EWMA state for the current window
                save_ewma(current_window_id, ewma_scores)

                logger.info("[EWMA] window %s alpha=%.4f", current_window_id, alpha)
                for uid_str, ew in ewma_scores.items():
                    uid_int = int(uid_str)
                    logger.debug(
                        "[EWMA] uid=%s ewma=%.4f current=%.4f prev=%s",
                        uid_int,
                        ew,
                        current_scores.get(uid_int, 0.0),
                        prev_scores.get(uid_str),
                    )

                # Use EWMA scores as the actual weights
                uids = [int(uid) for uid in ewma_scores.keys()]
                weights = list(ewma_scores.values())

                if not uids:
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_uids").inc()
                    last_done = block
                    continue

                ok = await retry_set_weights(wallet, uids, weights)
                if ok:
                    LASTSET_GAUGE.set(time.time())
                    VALIDATOR_LOOP_TOTAL.labels(outcome="success").inc()
                    VALIDATOR_LAST_BLOCK_SUCCESS.set(block)
                    loop_outcome = "success"
                    logger.info("set_weights OK at block %d", block)
                else:
                    logger.warning("set_weights failed at block %d", block)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="set_weights_failed").inc()
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    loop_outcome = "set_weights_failed"

                # ------------------------------
                # Update cache metrics
                # ------------------------------
                try:
                    sz = sum(
                        f.stat().st_size
                        for f in CACHE_DIR.glob("*.jsonl")
                        if f.is_file()
                    )
                    CACHE_FILES.set(len(list(CACHE_DIR.glob("*.jsonl"))))
                    VALIDATOR_CACHE_BYTES.set(sz)
                except Exception:
                    pass

                gc.collect()
                last_done = block

            except asyncio.CancelledError:
                raise
            except Exception as e:
                traceback.print_exc()
                logger.warning("Validator loop error: %s — reconnecting…", e)
                VALIDATOR_LOOP_TOTAL.labels(outcome="error").inc()
                loop_outcome = "error"
                st = None
                reset_subtensor()
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue
            finally:
                duration = asyncio.get_running_loop().time() - iter_start
                VALIDATOR_LAST_LOOP_DURATION_SECONDS.set(duration)

        except asyncio.CancelledError:
            break
        except Exception as e:
            traceback.print_exc()
            logger.warning("Validator loop error: %s — reconnecting…", e)
            VALIDATOR_LOOP_TOTAL.labels(outcome="error").inc()
            st = None
            reset_subtensor()
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("Validator shutting down gracefully...")
