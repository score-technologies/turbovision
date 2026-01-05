import os
import time
import asyncio
import logging
import signal
import traceback
import gc
from json import loads
from collections import defaultdict, deque
from functools import lru_cache
from aiohttp import ClientSession, ClientTimeout

import aiohttp
import bittensor as bt
from statistics import median

from scorevision.utils.cloudflare_helpers import (
    dataset_sv,
    dataset_sv_multi,
    ensure_index_exists,
    build_public_index_url,
    prune_sv,
)
from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    reset_subtensor,
    get_validator_indexes_from_chain,
    on_chain_commit_validator,
    _already_committed_same_index,
    _first_commit_block_by_miner,
)
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
    VALIDATOR_RECENT_WINDOW_SAMPLES,
    VALIDATOR_COMMIT_TOTAL,
    VALIDATOR_LAST_LOOP_DURATION_SECONDS,
    VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS,
)
from scorevision.utils.settings import get_settings

logger = logging.getLogger("scorevision.validator")

# Global shutdown event for graceful shutdown
shutdown_event = asyncio.Event()

for noisy in ["websockets", "websockets.client", "substrateinterface", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def _validator_hotkey_ss58() -> str:
    settings = get_settings()
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return wallet.hotkey.ss58_address


async def _validate_main(tail: int, alpha: float, m_min: int, tempo: int):
    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID
    R2_BUCKET_PUBLIC_URL = settings.R2_BUCKET_PUBLIC_URL

    def signal_handler():
        logger.info("Received shutdown signal, stopping validator...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    if os.getenv("SCOREVISION_COMMIT_VALIDATOR_ON_START", "1") not in (
        "0",
        "false",
        "False",
    ):
        try:
            index_url = None
            if R2_BUCKET_PUBLIC_URL:
                from scorevision.utils.cloudflare_helpers import (
                    build_public_index_url_from_public_base,
                )

                index_url = build_public_index_url_from_public_base(
                    R2_BUCKET_PUBLIC_URL
                )
            if not index_url:
                from scorevision.utils.cloudflare_helpers import build_public_index_url

                index_url = build_public_index_url()

            if not index_url:
                logger.warning(
                    "[validator-commit] No public index URL configured; skipping."
                )
                VALIDATOR_COMMIT_TOTAL.labels(result="no_index").inc()
            else:
                bootstrap_ok = True
                try:
                    await ensure_index_exists()
                except Exception as e:
                    bootstrap_ok = False
                    logger.warning(
                        "[validator-commit] ensure_index_exists failed (non-fatal bootstrap): %s",
                        e,
                    )

                force_bootstrap = os.getenv("VALIDATOR_BOOTSTRAP_COMMIT", "1") in (
                    "1",
                    "true",
                    "True",
                )
                if bootstrap_ok or force_bootstrap:
                    from scorevision.utils.bittensor_helpers import (
                        on_chain_commit_validator_retry,
                        _already_committed_same_index,
                    )

                    wait_blocks = int(os.getenv("VALIDATOR_COMMIT_WAIT_BLOCKS", "100"))
                    confirm_after = int(
                        os.getenv("VALIDATOR_COMMIT_CONFIRM_AFTER", "3")
                    )
                    max_retries_env = os.getenv("VALIDATOR_COMMIT_MAX_RETRIES")
                    max_retries = (
                        int(max_retries_env)
                        if (max_retries_env and max_retries_env.isdigit())
                        else None
                    )

                    same = await _already_committed_same_index(NETUID, index_url)
                    if same:
                        logger.info(
                            f"[validator-commit] Already published {index_url}; skipping."
                        )
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
                    logger.warning(
                        "[validator-commit] Skipping commit because ensure_index_exists failed and VALIDATOR_BOOTSTRAP_COMMIT is not set."
                    )
                    VALIDATOR_COMMIT_TOTAL.labels(result="no_index").inc()

        except Exception as e:
            logger.warning(f"[validator-commit] failed (non-fatal): {e}")
            VALIDATOR_COMMIT_TOTAL.labels(result="error").inc()

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
                uids, weights = await get_weights(tail=tail, m_min=m_min)
                if not uids:
                    logger.warning("No eligible uids this round; skipping.")
                    CURRENT_WINNER.set(-1)
                    VALIDATOR_WINNER_SCORE.set(0.0)
                    VALIDATOR_LOOP_TOTAL.labels(outcome="no_uids").inc()
                    loop_outcome = "no_uids"
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

                try:
                    await asyncio.to_thread(prune_sv, tail)
                except Exception as e:
                    logger.warning(f"Cache prune failed: {e}")

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
            # Check shutdown event during sleep
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("Validator shutting down gracefully...")


# ---------------- Weights selection (cross-validators) ---------------- #


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[1])
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(max(0.0, w) for _, w in pairs)
    if total <= 0:
        return median(values)
    acc = 0.0
    half = total / 2.0
    for v, w in pairs:
        acc += max(0.0, w)
        if acc >= half:
            return v
    return pairs[-1][0]


async def get_weights(tail: int = 36000, m_min: int = 25):
    """
    Cross-validators robust aggregation with stake-weighted outlier filtering.
    Uses alpha stake (meta.S) as the canonical stake for validators, with a
    fallback to legacy meta.stake if needed. Returns a single winner (uid, 1.0).
    """
    settings = get_settings()
    st = await get_subtensor()
    NETUID = settings.SCOREVISION_NETUID
    MECHID = settings.SCOREVISION_MECHID
    meta = await st.metagraph(NETUID, mechid=MECHID)
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys)}

    stake_tensor = getattr(meta, "S", None)
    if stake_tensor is None:
        stake_tensor = getattr(meta, "stake", None)

    stake_by_hk: dict[str, float] = {}
    if stake_tensor is not None:
        for i, hk in enumerate(meta.hotkeys):
            try:
                val = stake_tensor[i]
                if hasattr(val, "item"):
                    val = float(val.item())
                else:
                    val = float(val)
            except Exception:
                val = 0.0
            stake_by_hk[hk] = max(0.0, val)
    else:
        for hk in meta.hotkeys:
            stake_by_hk[hk] = 0.0

    validator_indexes = await get_validator_indexes_from_chain(NETUID)
    if not validator_indexes:
        logger.warning(
            "No validator registry found on-chain; falling back to local-only averaging."
        )

        sums: dict[str, float] = {}
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
            sums[hk] = sums.get(hk, 0.0) + score
            cnt[hk] = cnt.get(hk, 0) + 1

        if not cnt:
            logger.warning("No data in window → default SO")
            VALIDATOR_MINERS_CONSIDERED.set(0)
            return [6], [65535]
        elig = [
            hk for hk, n in cnt.items() if n >= m_min and hk in sums and hk in hk_to_uid
        ]
        if not elig:
            logger.warning("No hotkey reached %d samples → default SO", m_min)
            VALIDATOR_MINERS_CONSIDERED.set(0)
            return [6], [65535]
        avg = {hk: (sums[hk] / cnt[hk]) for hk in elig}
        VALIDATOR_MINERS_CONSIDERED.set(len(elig))
        winner_hk = max(avg, key=avg.get)
        CURRENT_WINNER.set(hk_to_uid[winner_hk])
        VALIDATOR_WINNER_SCORE.set(avg.get(winner_hk, 0.0))
        return [hk_to_uid[winner_hk]], [65535]

    # =========================
    # Cross-validators pipeline
    # =========================
    sums_by_V_m: dict[tuple[str, int], float] = {}
    cnt_by_V_m: dict[tuple[str, int], int] = {}

    async for line in dataset_sv_multi(tail, validator_indexes):
        try:
            payload = line.get("payload") or {}
            miner = payload.get("miner") or {}
            miner_hk = (miner.get("hotkey") or "").strip()
            if not miner_hk or miner_hk not in hk_to_uid:
                continue
            miner_uid = hk_to_uid[miner_hk]
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
            score = float(((payload.get("evaluation") or {}).get("score")) or 0.0)
        except Exception:
            continue

        key = (validator_hk, miner_uid)
        sums_by_V_m[key] = sums_by_V_m.get(key, 0.0) + score
        cnt_by_V_m[key] = cnt_by_V_m.get(key, 0) + 1

    if not cnt_by_V_m:
        logger.warning("No cross-validator data in window → default SO")
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return [6], [65535]

    mu_by_V_m: dict[tuple[str, int], tuple[float, int]] = {}
    for key, n in cnt_by_V_m.items():
        s = sums_by_V_m.get(key, 0.0)
        mu_by_V_m[key] = (s / max(1, n), n)
    logger.info(
        "Validator→Miner means: "
        + ", ".join(
            f"{V}->{m}: μ={mu:.4f} (n={n})" for (V, m), (mu, n) in mu_by_V_m.items()
        )
    )

    a_rob, b_rob = 0.5, 0.5
    k = 2.5
    eps = 1e-3
    a_final, b_final = 1.0, 0.5

    miners_seen = set([m for (_V, m) in mu_by_V_m.keys()])
    S_by_m: dict[int, float] = {}

    for m in miners_seen:
        mus: list[float] = []
        wtilde: list[float] = []
        triplets: list[tuple[str, float, int]] = []

        for (V, mm), (mu, n) in mu_by_V_m.items():
            if mm != m:
                continue
            if n < m_min:
                continue
            stake = stake_by_hk.get(V, 0.0)
            wt = (stake**a_rob) * ((max(1, n)) ** b_rob)
            mus.append(mu)
            wtilde.append(wt)
            triplets.append((V, mu, n))

        if not mus or sum(max(0.0, w) for w in wtilde) <= 0:
            continue

        med = _weighted_median(mus, wtilde)
        abs_dev = [abs(x - med) for x in mus]
        MAD = _weighted_median(abs_dev, wtilde)
        if MAD < eps:
            MAD = eps
        thresh = k * (MAD / 0.6745)

        filtered: list[tuple[str, float, int]] = []
        rejected: list[tuple[str, float, int]] = []
        for V, mu, n in triplets:
            if abs(mu - med) <= thresh:
                filtered.append((V, mu, n))
            else:
                rejected.append((V, mu, n))
        if rejected:
            logger.info(
                f"Miner {m}: rejected {len(rejected)} validator means (outliers) → "
                + ", ".join(f"{V}: μ={mu:.4f} (n={n})" for V, mu, n in rejected)
            )
        if len(filtered) < 2:
            VALIDATOR_MINERS_SKIPPED_TOTAL.labels(reason="insufficient_filtered").inc()
            continue

        num = 0.0
        den = 0.0
        for V, mu, n in filtered:
            stake = stake_by_hk.get(V, 0.0)
            wf = (stake**a_final) * ((max(1, n)) ** b_final)
            num += wf * mu
            den += wf
        if den <= 0:
            continue
        S_by_m[m] = num / den

    validator_uid = None
    try:
        validator_uid = hk_to_uid.get(_validator_hotkey_ss58())
    except Exception:
        validator_uid = None

    if validator_uid is not None and validator_uid in S_by_m:
        logger.info(
            "Excluding validator uid=%d from weight candidates to avoid self-weight.",
            validator_uid,
        )
        S_by_m.pop(validator_uid, None)

    if not S_by_m:
        logger.warning("No miners passed robust filtering.")
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return [6], [65535]

    VALIDATOR_MINERS_CONSIDERED.set(len(S_by_m))

    logger.info(
        "Final miner means: "
        + ", ".join(f"uid={m}: {s:.4f}" for m, s in sorted(S_by_m.items()))
    )

    winner_uid = max(S_by_m, key=S_by_m.get)
    logger.info(
        "Provisional winner uid=%d S=%.4f over last %d blocks",
        winner_uid,
        S_by_m[winner_uid],
        tail,
    )
    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(S_by_m.get(winner_uid, 0.0))

    if settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE:
        try:
            mu_recent_by_V_m, n_total_recent_by_m = await _collect_recent_mu_by_V_m(
                tail=tail,
                validator_indexes=validator_indexes,
                hk_to_uid=hk_to_uid,
                K=settings.SCOREVISION_WINDOW_K_PER_VALIDATOR,
            )

            if mu_recent_by_V_m:
                S_recent = _aggregate_recent_S_by_m(
                    mu_recent_by_V_m,
                    stake_by_hk=stake_by_hk,
                    a_final=a_final,
                    b_final=b_final,
                )

                uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
                first_commit_block_by_hk = await _first_commit_block_by_miner(NETUID)

                final_uid = _pick_winner_with_window_tiebreak(
                    winner_uid,
                    hk_to_uid=hk_to_uid,
                    uid_to_hk=uid_to_hk,
                    S_recent=S_recent,
                    delta_abs=settings.SCOREVISION_WINDOW_DELTA_ABS,
                    delta_rel=settings.SCOREVISION_WINDOW_DELTA_REL,
                    first_commit_block_by_hk=first_commit_block_by_hk,
                )

                if final_uid != winner_uid:
                    logger.info(
                        "[window-tiebreak] Provisional winner=%d (S_recent=%.6f) -> "
                        "Final winner=%d (S_recent=%.6f) via earliest on-chain commit",
                        winner_uid,
                        S_recent.get(winner_uid, float("nan")),
                        final_uid,
                        S_recent.get(final_uid, float("nan")),
                    )
                    winner_uid = final_uid
            else:
                logger.info(
                    "[window-tiebreak] No recent data available; keeping provisional winner."
                )

        except Exception as e:
            logger.warning(
                f"[window-tiebreak] disabled due to error: {type(e).__name__}: {e}"
            )

    logger.info(
        "Winner uid=%d (after window tie-break) over last %d blocks", winner_uid, tail
    )

    TARGET_UID = 6
    final_score = float(S_by_m.get(winner_uid, 0.0) or 0.0)

    if abs(final_score) <= 1e-12:
        logger.info(
            "Final winner uid=%d has final_score=%.6f -> forcing winner to TARGET_UID=%d",
            winner_uid,
            final_score,
            TARGET_UID,
        )
        winner_uid = TARGET_UID
        final_score = float(S_by_m.get(TARGET_UID, 0.0) or 0.0)

    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(final_score)

    if winner_uid == TARGET_UID:
        return [TARGET_UID], [1.0]
    return [winner_uid, TARGET_UID], [0.10, 0.90]


async def retry_set_weights(wallet, uids, weights):

    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID
    MECHID = settings.SCOREVISION_MECHID
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


async def _collect_recent_mu_by_V_m(
    tail: int,
    validator_indexes: dict[str, str],
    hk_to_uid: dict[str, int],
    *,
    K: int = 25,
) -> tuple[dict[tuple[str, int], tuple[float, int]], dict[int, int]]:
    """ """
    deques_by_V_m: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=K))

    async for line in dataset_sv_multi(tail, validator_indexes):
        try:
            payload = line.get("payload") or {}
            miner = payload.get("miner") or {}
            miner_hk = (miner.get("hotkey") or "").strip()
            if not miner_hk or miner_hk not in hk_to_uid:
                continue
            m = hk_to_uid[miner_hk]
            V = (line.get("hotkey") or "").strip()
            if not V:
                continue
            score = float(((payload.get("evaluation") or {}).get("score")) or 0.0)
        except Exception:
            continue
        deques_by_V_m[(V, m)].append(score)

    mu_recent_by_V_m: dict[tuple[str, int], tuple[float, int]] = {}
    n_total_recent_by_m: dict[int, int] = defaultdict(int)
    for key, dq in deques_by_V_m.items():
        if not dq:
            continue
        mu = sum(dq) / len(dq)
        n = len(dq)
        mu_recent_by_V_m[key] = (mu, n)
        n_total_recent_by_m[key[1]] += n

    for uid, total in n_total_recent_by_m.items():
        VALIDATOR_RECENT_WINDOW_SAMPLES.labels(uid=str(uid)).set(total)

    return mu_recent_by_V_m, n_total_recent_by_m


def _stake_of(hk: str, stake_by_hk: dict[str, float]) -> float:
    try:
        return max(0.0, float(stake_by_hk.get(hk, 0.0)))
    except Exception:
        return 0.0


def _aggregate_recent_S_by_m(
    mu_recent_by_V_m: dict[tuple[str, int], tuple[float, int]],
    stake_by_hk: dict[str, float],
    *,
    a_final: float = 1.0,
    b_final: float = 0.5,
) -> dict[int, float]:
    """ """
    num_by_m: dict[int, float] = defaultdict(float)
    den_by_m: dict[int, float] = defaultdict(float)

    for (V, m), (mu, n) in mu_recent_by_V_m.items():
        stake = _stake_of(V, stake_by_hk)
        wf = (stake**a_final) * ((max(1, n)) ** b_final)
        num_by_m[m] += wf * mu
        den_by_m[m] += wf

    S_recent: dict[int, float] = {}
    for m, den in den_by_m.items():
        if den > 0:
            S_recent[m] = num_by_m[m] / den
    return S_recent


def _pick_winner_with_window_tiebreak(
    winner_uid: int,
    hk_to_uid: dict[str, int],
    uid_to_hk: dict[int, str],
    S_recent: dict[int, float],
    *,
    delta_abs: float,
    delta_rel: float,
    first_commit_block_by_hk: dict[str, int],
) -> int:
    """ """
    if winner_uid not in S_recent:
        return winner_uid

    s_win = S_recent[winner_uid]
    window_hi = s_win + max(delta_abs, delta_rel * abs(s_win))
    window_lo = s_win - max(delta_abs, delta_rel * abs(s_win))

    close_uids = [m for m, s in S_recent.items() if window_lo <= s <= window_hi]
    if winner_uid not in close_uids:
        close_uids.append(winner_uid)

    if len(close_uids) == 1:
        return winner_uid

    best_uid = winner_uid
    best_blk = None

    for m in close_uids:
        hk = uid_to_hk.get(m)
        if not hk:
            continue
        blk = first_commit_block_by_hk.get(hk)
        if blk is None:
            candidate = 10**18
        else:
            candidate = int(blk)

        if (
            (best_blk is None)
            or (candidate < best_blk)
            or (candidate == best_blk and hk < uid_to_hk.get(best_uid, ""))
        ):
            best_blk = candidate
            best_uid = m

    return best_uid
