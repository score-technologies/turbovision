import asyncio
import gc
import math
import os
import signal
import time
import traceback
from functools import lru_cache
from logging import getLogger
from pathlib import Path
import aiohttp
import bittensor as bt
from scorevision.utils.settings import get_settings
from scorevision.utils.windows import get_current_window_id
from scorevision.utils.prometheus import (
    LASTSET_GAUGE,
    CACHE_DIR,
    CACHE_FILES,
    CURRENT_WINNER,
    VALIDATOR_BLOCK_HEIGHT,
    VALIDATOR_LOOP_TOTAL,
    VALIDATOR_LAST_BLOCK_SUCCESS,
    VALIDATOR_WEIGHT_FAIL_TOTAL,
    VALIDATOR_CACHE_BYTES,
    VALIDATOR_WINNER_SCORE,
    VALIDATOR_COMMIT_TOTAL,
    VALIDATOR_LAST_LOOP_DURATION_SECONDS,
    VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS,
)
from scorevision.utils.bittensor_helpers import (
    reset_subtensor,
    get_subtensor,
    on_chain_commit_validator_retry,
    _already_committed_same_index,
)
from scorevision.utils.blacklist import load_blacklisted_hotkeys
from scorevision.utils.cloudflare_helpers import (
    ensure_index_exists,
    build_public_index_url_from_public_base,
    prune_sv,
    put_winners_snapshot,
)
from scorevision.utils.manifest import (
    get_current_manifest,
    Manifest,
    load_manifest_from_public_index,
)
from scorevision.validator.payload import extract_elements_from_manifest
from scorevision.validator.scoring import days_to_blocks
from scorevision.validator.winner import get_winner_for_element

logger = getLogger(__name__)
shutdown_event = asyncio.Event()


@lru_cache(maxsize=1)
def get_validator_hotkey_ss58() -> str:
    settings = get_settings()
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return wallet.hotkey.ss58_address


async def set_weights_via_signer(wallet, uids: list[int], weights: list[float]) -> bool:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID
    mechid = settings.SCOREVISION_MECHID
    signer_url = settings.SIGNER_URL

    loop = asyncio.get_running_loop()
    request_start = loop.time()
    try:
        logger.info("SETTING WEIGHTS uids=%s weights=%s", uids, weights)
        timeout = aiohttp.ClientTimeout(connect=2, total=300)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": netuid,
                    "mechid": mechid,
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
                body_txt = data if isinstance(data, str) else (data.get("error") or data.get("raw") or "")
            except Exception:
                pass
            if "SettingWeightsTooFast" in str(body_txt):
                logger.warning("Signer returns SettingWeightsTooFast; weights likely set.")
                return True

            VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_http").inc()
            logger.warning("Signer error status=%s body=%s", resp.status, data)

    except aiohttp.ClientConnectorError as e:
        VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS.set(loop.time() - request_start)
        logger.warning("Signer unreachable: %s", e)
        VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_connect").inc()
    except asyncio.TimeoutError:
        VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS.set(loop.time() - request_start)
        logger.warning("Signer timed out")
        VALIDATOR_WEIGHT_FAIL_TOTAL.labels(stage="signer_timeout").inc()

    return False


async def load_manifest_for_block(block: int, *, path_manifest: Path | None = None) -> Manifest:
    settings = get_settings()
    if getattr(settings, "URL_MANIFEST", None):
        cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
        return await load_manifest_from_public_index(
            settings.URL_MANIFEST,
            block_number=block,
            cache_dir=cache_dir,
        )
    if path_manifest is not None:
        return Manifest.load_yaml(path_manifest)
    p = (
        os.getenv("SCOREVISION_MANIFEST_PATH")
        or os.getenv("SV_MANIFEST_PATH")
        or os.getenv("SCOREVISION_VALIDATOR_MANIFEST_PATH")
    )
    if p:
        return Manifest.load_yaml(Path(p))
    return get_current_manifest(block_number=block)


def setup_signal_handler():
    def handler():
        logger.warning("Received shutdown signal, stopping weights loop...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: handler())


async def commit_validator_on_start(netuid: int):
    settings = get_settings()
    r2_bucket_public_url = settings.SCOREVISION_PUBLIC_RESULTS_URL

    if os.getenv("SCOREVISION_COMMIT_VALIDATOR_ON_START", "1") in ("0", "false", "False"):
        return

    try:
        index_url = build_public_index_url_from_public_base(r2_bucket_public_url)

        if not index_url:
            logger.warning("[validator-commit] SCOREVISION_PUBLIC_RESULTS_URL is not set; skipping.")
            VALIDATOR_COMMIT_TOTAL.labels(result="no_index").inc()
            return

        bootstrap_ok = True
        try:
            await ensure_index_exists()
        except Exception as e:
            bootstrap_ok = False
            logger.warning("[validator-commit] ensure_index_exists failed: %s", e)

        force_bootstrap = os.getenv("VALIDATOR_BOOTSTRAP_COMMIT", "1") in ("1", "true", "True")
        if bootstrap_ok or force_bootstrap:
            wait_blocks = int(os.getenv("VALIDATOR_COMMIT_WAIT_BLOCKS", "100"))
            confirm_after = int(os.getenv("VALIDATOR_COMMIT_CONFIRM_AFTER", "3"))
            max_retries_env = os.getenv("VALIDATOR_COMMIT_MAX_RETRIES")
            max_retries = int(max_retries_env) if (max_retries_env and max_retries_env.isdigit()) else None

            same = await _already_committed_same_index(netuid, index_url)
            if same:
                logger.info("[validator-commit] Already published %s; skipping.", index_url)
                VALIDATOR_COMMIT_TOTAL.labels(result="already_published").inc()
            else:
                ok = await on_chain_commit_validator_retry(
                    index_url,
                    wait_blocks=wait_blocks,
                    confirm_after=confirm_after,
                    max_retries=max_retries,
                )
                if ok:
                    VALIDATOR_COMMIT_TOTAL.labels(result="committed").inc()
                else:
                    VALIDATOR_COMMIT_TOTAL.labels(result="error").inc()
        else:
            logger.warning("[validator-commit] Skipping commit; ensure_index_exists failed.")
            VALIDATOR_COMMIT_TOTAL.labels(result="no_index").inc()

    except Exception as e:
        logger.warning("[validator-commit] failed (non-fatal): %s", e)
        VALIDATOR_COMMIT_TOTAL.labels(result="error").inc()


async def weights_loop(
    tail: int = 28800,
    m_min: int = 25,
    tempo: int = 150,
    path_manifest: Path | None = None,
    commit_on_start: bool = False,
) -> None:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    tail_blocks_default = settings.VALIDATOR_TAIL_BLOCKS
    winners_every_n = settings.VALIDATOR_WINNERS_EVERY_N

    setup_signal_handler()
    if commit_on_start:
        await commit_validator_on_start(netuid)

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    subtensor = None
    last_done = -1
    effective_tail = max(tail, tail_blocks_default)
    set_weights_count = 0
    validator_hotkey_ss58 = get_validator_hotkey_ss58()
    central_validator_hotkey = (settings.SCOREVISION_CENTRAL_VALIDATOR_HOTKEY or "").strip()
    is_central_validator = bool(central_validator_hotkey) and validator_hotkey_ss58 == central_validator_hotkey
    if not is_central_validator:
        logger.info(
            "[weights] winners snapshots disabled for hotkey=%s (central=%s)",
            validator_hotkey_ss58,
            central_validator_hotkey or "<unset>",
        )

    while not shutdown_event.is_set():
        try:
            blacklisted_hotkeys = load_blacklisted_hotkeys()
            if blacklisted_hotkeys:
                logger.info("[weights] loaded %d blacklisted hotkeys", len(blacklisted_hotkeys))

            if subtensor is None:
                subtensor = await get_subtensor()

            block = await subtensor.get_current_block()
            VALIDATOR_BLOCK_HEIGHT.set(block)

            current_window_id = get_current_window_id(block, tempo=tempo)
            logger.info("[weights] current_window_id=%s (block=%d, tempo=%d)", current_window_id, block, tempo)

            if block % tempo != 0 or block <= last_done:
                try:
                    await asyncio.wait_for(subtensor.wait_for_block(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning("wait_for_block error (%s); resetting subtensor", err)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="subtensor_error").inc()
                    reset_subtensor()
                    subtensor = None
                    await asyncio.sleep(2.0)
                    continue
                continue

            iter_loop = asyncio.get_running_loop()
            iter_start = iter_loop.time()

            try:
                try:
                    manifest = await load_manifest_for_block(block, path_manifest=path_manifest)
                except Exception as e:
                    logger.warning("[weights] Failed to load manifest for block %d: %s", block, e)
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="manifest_error").inc()
                    last_done = block
                    try:
                        await asyncio.wait_for(subtensor.wait_for_block(), timeout=30.0)
                    except Exception:
                        pass
                    continue

                elements = extract_elements_from_manifest(manifest)
                if not elements:
                    logger.warning("[weights] Manifest has no elements for window_id=%s", getattr(manifest, "window_id", None))
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_elements").inc()
                    last_done = block
                    continue

                weights_by_uid: dict[int, float] = {}
                winners_by_element: dict[str, dict[str, str | None]] = {}
                max_tail_used = effective_tail

                total_elem_weight = sum(max(0.0, w) for _eid, w, _ew in elements)
                if total_elem_weight <= 0:
                    logger.warning("[weights] Element weights sum to 0 -> forcing fallback_uid=%d", fallback_uid)
                    weights_by_uid = {fallback_uid: 1.0}
                else:
                    elements = [(eid, max(0.0, float(w)), eval_window_days) for (eid, w, eval_window_days) in elements]
                    max_tail_used = 0

                    for element_id, elem_weight, eval_window_days in elements:
                        tail_from_eval = days_to_blocks(eval_window_days)
                        tail_for_element = tail_from_eval if tail_from_eval is not None else effective_tail
                        max_tail_used = max(max_tail_used, tail_for_element)

                        logger.info(
                            "[weights] element=%s eval_window_days=%s -> tail_blocks=%d",
                            element_id,
                            str(eval_window_days),
                            tail_for_element,
                        )

                        winner_uid, scores_by_miner, winner_meta = await get_winner_for_element(
                            element_id=element_id,
                            current_window_id=current_window_id,
                            tail=tail_for_element,
                            m_min=m_min,
                            blacklisted_hotkeys=blacklisted_hotkeys,
                            validator_hotkey_ss58=validator_hotkey_ss58,
                        )

                        if winner_uid is None:
                            logger.warning("[weights] No winner for element_id=%s", element_id)
                            continue

                        share = float(elem_weight)
                        weights_by_uid[winner_uid] = weights_by_uid.get(winner_uid, 0.0) + share

                        logger.info(
                            "[weights] Element=%s winner_uid=%d elem_weight=%.6f",
                            element_id,
                            winner_uid,
                            elem_weight,
                        )
                        if winner_meta and winner_meta.get("hotkey"):
                            winners_by_element[element_id] = {
                                "winner_hotkey": winner_meta.get("hotkey"),
                                "chute_id": winner_meta.get("chute_id"),
                                "slug": winner_meta.get("slug"),
                            }

                if blacklisted_hotkeys:
                    try:
                        meta = await subtensor.metagraph(netuid, mechid=settings.SCOREVISION_MECHID)
                        uid_to_hk = {i: hk for i, hk in enumerate(meta.hotkeys)}
                        for uid in list(weights_by_uid.keys()):
                            hk = uid_to_hk.get(uid)
                            if hk and hk in blacklisted_hotkeys:
                                logger.info("[weights] Removing blacklisted uid=%d", uid)
                                weights_by_uid.pop(uid, None)
                    except Exception as e:
                        logger.warning("[weights] Failed to apply blacklist: %s", e)

                if not weights_by_uid:
                    logger.warning("[weights] No winners found for window_id=%s", current_window_id)
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_uids").inc()
                    last_done = block
                    continue

                total_weight = sum(weights_by_uid.values())
                if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    if total_weight > 1.0:
                        logger.warning("[weights] Total weights %.6f > 1.0; normalizing", total_weight)
                        scale = 1.0 / total_weight
                        for uid in list(weights_by_uid.keys()):
                            weights_by_uid[uid] *= scale
                    else:
                        missing = 1.0 - total_weight
                        weights_by_uid[fallback_uid] = weights_by_uid.get(fallback_uid, 0.0) + missing
                        logger.info("[weights] Total weights %.6f < 1.0; adding fallback", total_weight)

                uids = sorted(weights_by_uid.keys())
                weights = [weights_by_uid[uid] for uid in uids]

                logger.info(
                    "[weights] Final weights for window_id=%s: %s",
                    current_window_id,
                    ", ".join(f"uid={u}: w={w:.6f}" for u, w in zip(uids, weights)),
                )

                ok = await set_weights_via_signer(wallet, uids, weights)
                if ok:
                    LASTSET_GAUGE.set(time.time())
                    VALIDATOR_LOOP_TOTAL.labels(outcome="success").inc()
                    VALIDATOR_LAST_BLOCK_SUCCESS.set(block)
                    logger.info("set_weights OK at block %d", block)
                    set_weights_count += 1
                    if winners_every_n > 0 and set_weights_count % winners_every_n == 0:
                        if is_central_validator and winners_by_element:
                            payload = {
                                "block": block,
                                "window_id": current_window_id,
                                "netuid": settings.SCOREVISION_NETUID,
                                "mechid": settings.SCOREVISION_MECHID,
                                "winners": winners_by_element,
                            }
                            timeout_s = float(os.getenv("SV_R2_TIMEOUT_S", "60"))
                            try:
                                key = await asyncio.wait_for(put_winners_snapshot(block, payload), timeout=timeout_s)
                                logger.info("[weights] winners snapshot stored: %s", key)
                            except asyncio.TimeoutError:
                                logger.warning("[weights] winners snapshot timed out")
                            except Exception as e:
                                logger.warning("[weights] winners snapshot failed: %s", e)
                else:
                    logger.warning("set_weights failed at block %d", block)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="set_weights_failed").inc()
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)

                try:
                    sz = sum(f.stat().st_size for f in CACHE_DIR.glob("*.jsonl") if f.is_file())
                    CACHE_FILES.set(len(list(CACHE_DIR.glob("*.jsonl"))))
                    VALIDATOR_CACHE_BYTES.set(sz)
                except Exception:
                    pass

                try:
                    prune_tail = max(max_tail_used, effective_tail)
                    await asyncio.to_thread(prune_sv, prune_tail)
                except Exception as e:
                    logger.warning("Cache prune failed: %s", e)

                gc.collect()
                last_done = block

            except asyncio.CancelledError:
                raise
            except Exception as e:
                traceback.print_exc()
                logger.warning("Weights loop error: %s", e)
                VALIDATOR_LOOP_TOTAL.labels(outcome="error").inc()
                subtensor = None
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
            logger.warning("Weights loop error: %s", e)
            VALIDATOR_LOOP_TOTAL.labels(outcome="error").inc()
            subtensor = None
            reset_subtensor()
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("Weights loop shutting down gracefully...")
