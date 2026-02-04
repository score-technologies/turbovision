import asyncio
import gc
import math
import os
import signal
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from statistics import median
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
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
    VALIDATOR_COMMIT_TOTAL,
    VALIDATOR_LAST_LOOP_DURATION_SECONDS,
    VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS,
)
from scorevision.utils.bittensor_helpers import (
    reset_subtensor,
    get_subtensor,
    get_validator_indexes_from_chain,
    on_chain_commit_validator_retry,
    _already_committed_same_index,
    _first_commit_block_by_miner,
)
from scorevision.utils.blacklist import load_blacklisted_hotkeys
from scorevision.utils.cloudflare_helpers import (
    dataset_sv,
    dataset_sv_multi,
    ensure_index_exists,
    build_public_index_url,
    prune_sv,
    put_winners_snapshot,
)
from scorevision.utils.manifest import (
    get_current_manifest,
    Manifest,
    load_manifest_from_public_index,
)

logger = getLogger(__name__)
shutdown_event = asyncio.Event()


@dataclass
class WeightsResult:
    element_id: str
    window_id: str
    winner_uid: int | None
    scores_by_uid: dict[int, float]
    winner_meta: dict[str, str | None] | None


@lru_cache(maxsize=1)
def get_validator_hotkey_ss58() -> str:
    settings = get_settings()
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return wallet.hotkey.ss58_address


def extract_miner_and_score_from_payload(payload: dict, hk_to_uid: dict[str, int]):
    try:
        telemetry = payload.get("telemetry") or {}
        miner_info = telemetry.get("miner") or {}
        miner_hk = (miner_info.get("hotkey") or "").strip()
        if not miner_hk or miner_hk not in hk_to_uid:
            return None, None
        metrics = payload.get("metrics") or {}
        score = metrics.get("composite_score", payload.get("composite_score", 0.0))
        score = float(score)
        miner_uid = hk_to_uid[miner_hk]
        return miner_uid, score
    except Exception:
        return None, None


def extract_miner_meta_from_payload(payload: dict) -> dict[str, str | None] | None:
    try:
        telemetry = payload.get("telemetry") or {}
        miner_info = telemetry.get("miner") or {}
        miner_hk = (miner_info.get("hotkey") or "").strip()
        if not miner_hk:
            return None
        return {
            "hotkey": miner_hk,
            "chute_id": miner_info.get("chute_id"),
            "slug": miner_info.get("slug"),
        }
    except Exception:
        return None


def build_winner_meta_from_uid(
    winner_uid: int | None,
    uid_to_hk: dict[int, str],
    miner_meta_by_hk: dict[str, dict[str, str | None]],
) -> dict[str, str | None] | None:
    if winner_uid is None:
        return None
    winner_hk = uid_to_hk.get(winner_uid)
    if not winner_hk:
        return None
    meta = miner_meta_by_hk.get(winner_hk) or {}
    return {
        "hotkey": winner_hk,
        "chute_id": meta.get("chute_id"),
        "slug": meta.get("slug"),
    }


def weighted_median(values: list[float], weights: list[float]) -> float:
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


def days_to_blocks(days: int | float | None) -> int | None:
    if days is None:
        return None
    try:
        d = float(days)
    except Exception:
        return None
    if d <= 0:
        return None
    settings = get_settings()
    return max(1, int(d * settings.BLOCKS_PER_DAY))


def stake_of(hk: str, stake_by_hk: dict[str, float]) -> float:
    try:
        return max(0.0, float(stake_by_hk.get(hk, 0.0)))
    except Exception:
        return 0.0


def extract_challenge_id_from_payload(payload: dict) -> str | None:
    meta = payload.get("meta") or {}
    telemetry = payload.get("telemetry") or {}
    cand = (
        meta.get("task_id")
        or payload.get("task_id")
        or telemetry.get("task_id")
        or meta.get("challenge_id")
        or payload.get("challenge_id")
        or telemetry.get("challenge_id")
        or payload.get("job_id")
        or telemetry.get("job_id")
    )
    if cand is None:
        return None
    try:
        s = str(cand).strip()
        return s or None
    except Exception:
        return None


def extract_elements_from_manifest(manifest) -> list[tuple[str, float, int | float | None]]:
    elements = getattr(manifest, "elements", None) or []
    out: list[tuple[str, float, int | float | None]] = []
    for elem in elements:
        eid = None
        weight = None
        eval_window = None
        if hasattr(elem, "element_id"):
            eid = getattr(elem, "element_id")
        elif hasattr(elem, "id"):
            eid = getattr(elem, "id")
        elif isinstance(elem, dict):
            eid = elem.get("element_id") or elem.get("id")
        if hasattr(elem, "weight"):
            weight = getattr(elem, "weight")
        elif isinstance(elem, dict):
            weight = elem.get("weight")
        if hasattr(elem, "eval_window"):
            eval_window = getattr(elem, "eval_window")
        elif isinstance(elem, dict):
            eval_window = elem.get("eval_window")
        if eid is None:
            continue
        try:
            eid_str = str(eid)
        except Exception:
            continue
        try:
            w = float(weight) if weight is not None else 0.0
        except Exception:
            w = 0.0
        ew = None
        if eval_window is not None:
            try:
                ew = float(eval_window)
                if ew.is_integer():
                    ew = int(ew)
            except Exception:
                ew = None
        out.append((eid_str, w, ew))
    return out


def are_similar_by_challenges(
    challenge_scores1: dict[str, float],
    challenge_scores2: dict[str, float],
    *,
    delta_abs: float,
    delta_rel: float,
    min_common_challenges: int = 5,
) -> bool:
    common = []
    all_ids = set(challenge_scores1.keys()) | set(challenge_scores2.keys())
    for cid in all_ids:
        s1 = float(challenge_scores1.get(cid, 0.0) or 0.0)
        s2 = float(challenge_scores2.get(cid, 0.0) or 0.0)
        if abs(s1) > 1e-9 and abs(s2) > 1e-9:
            common.append((cid, s1, s2))
    if len(common) < min_common_challenges:
        return False
    for _cid, s1, s2 in common:
        max_score = max(abs(s1), abs(s2))
        thr = max(delta_abs, delta_rel * max_score)
        if abs(s1 - s2) > thr:
            return False
    return True


async def get_local_fallback_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    hk_to_uid: dict[str, int],
) -> tuple[int | None, dict[int, float], dict[str, str | None] | None]:
    settings = get_settings()
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    sums: dict[int, float] = {}
    cnt: dict[int, int] = {}
    miner_meta_by_hk: dict[str, dict[str, str | None]] = {}

    async for line in dataset_sv(tail):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score_from_payload(payload, hk_to_uid)
            if miner_uid is None:
                continue
            miner_meta = extract_miner_meta_from_payload(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta["hotkey"]] = miner_meta
        except Exception:
            continue
        sums[miner_uid] = sums.get(miner_uid, 0.0) + score
        cnt[miner_uid] = cnt.get(miner_uid, 0) + 1

    if not cnt:
        logger.warning(
            "[weights] No local data for element_id=%s window_id=%s -> fallback",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta_from_uid(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    elig = [uid for uid, n in cnt.items() if n >= m_min and uid in sums]
    if not elig:
        logger.warning(
            "[weights] No miner reached %d samples for element_id=%s -> fallback",
            m_min,
            element_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta_from_uid(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    avg = {uid: (sums[uid] / cnt[uid]) for uid in elig}
    VALIDATOR_MINERS_CONSIDERED.set(len(elig))
    winner_uid = max(avg, key=avg.get)
    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(avg.get(winner_uid, 0.0))
    uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
    return winner_uid, avg, build_winner_meta_from_uid(winner_uid, uid_to_hk, miner_meta_by_hk)


async def collect_recent_challenge_scores_by_validator_miner(
    tail: int,
    validator_indexes: dict[str, str],
    hk_to_uid: dict[str, int],
    *,
    element_id: str,
    current_window_id: str,
    K: int = 25,
) -> dict[tuple[str, int], deque]:
    challenge_scores: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=K))
    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score_from_payload(payload, hk_to_uid)
            if miner_uid is None:
                continue
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
            challenge_id = extract_challenge_id_from_payload(payload)
            if not challenge_id:
                continue
            challenge_key = f"{validator_hk}:{challenge_id}"
            challenge_scores[(validator_hk, miner_uid)].append((challenge_key, float(score)))
        except Exception:
            continue
    return challenge_scores


def aggregate_challenge_scores_by_miner(
    challenge_scores_by_validator_miner: dict[tuple[str, int], deque],
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for (_validator_hk, miner_uid), dq in challenge_scores_by_validator_miner.items():
        if miner_uid not in result:
            result[miner_uid] = {}
        for challenge_id, score in dq:
            result[miner_uid][challenge_id] = float(score)
    return result


def pick_winner_with_window_tiebreak(
    winner_uid: int,
    uid_to_hk: dict[int, str],
    challenge_scores_by_miner: dict[int, dict[str, float]],
    candidate_uids: set[int],
    *,
    delta_abs: float,
    delta_rel: float,
    first_commit_block_by_hk: dict[str, int],
    min_common_challenges: int = 5,
) -> int:
    if winner_uid not in challenge_scores_by_miner:
        return winner_uid
    winner_scores = challenge_scores_by_miner[winner_uid]
    similar_uids = [winner_uid]
    for miner_uid in candidate_uids:
        if miner_uid == winner_uid:
            continue
        scores = challenge_scores_by_miner.get(miner_uid)
        if not scores:
            continue
        if are_similar_by_challenges(
            winner_scores,
            scores,
            delta_abs=delta_abs,
            delta_rel=delta_rel,
            min_common_challenges=min_common_challenges,
        ):
            similar_uids.append(miner_uid)
    if len(similar_uids) == 1:
        return winner_uid
    best_uid = winner_uid
    best_blk = None
    for miner_uid in similar_uids:
        hk = uid_to_hk.get(miner_uid)
        if not hk:
            continue
        blk = first_commit_block_by_hk.get(hk)
        candidate = int(blk) if blk is not None else 10**18
        if (best_blk is None) or (candidate < best_blk):
            best_blk = candidate
            best_uid = miner_uid
    return best_uid


async def get_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    blacklisted_hotkeys: set[str] | None = None,
) -> tuple[int | None, dict[int, float], dict[str, str | None] | None]:
    settings = get_settings()
    subtensor = await get_subtensor()
    netuid = settings.SCOREVISION_NETUID
    mechid = settings.SCOREVISION_MECHID
    fallback_uid = settings.VALIDATOR_FALLBACK_UID

    meta = await subtensor.metagraph(netuid, mechid=mechid)
    if blacklisted_hotkeys is None:
        blacklisted_hotkeys = set()
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys) if hk not in blacklisted_hotkeys}
    uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}

    stake_tensor = getattr(meta, "S", None) or getattr(meta, "stake", None)
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

    validator_indexes = await get_validator_indexes_from_chain(netuid)
    if not validator_indexes:
        logger.warning(
            "[weights] No validator registry found on-chain for element_id=%s",
            element_id,
        )
        return await get_local_fallback_winner_for_element(
            element_id=element_id,
            current_window_id=current_window_id,
            tail=tail,
            m_min=m_min,
            hk_to_uid=hk_to_uid,
        )

    sums_by_validator_miner: dict[tuple[str, int], float] = {}
    cnt_by_validator_miner: dict[tuple[str, int], int] = {}
    miner_meta_by_hk: dict[str, dict[str, str | None]] = {}

    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score_from_payload(payload, hk_to_uid)
            if miner_uid is None:
                continue
            miner_meta = extract_miner_meta_from_payload(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta["hotkey"]] = miner_meta
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
        except Exception:
            continue
        key = (validator_hk, miner_uid)
        sums_by_validator_miner[key] = sums_by_validator_miner.get(key, 0.0) + score
        cnt_by_validator_miner[key] = cnt_by_validator_miner.get(key, 0) + 1

    if not cnt_by_validator_miner:
        logger.warning(
            "[weights] No cross-validator data for element_id=%s window_id=%s -> fallback",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta_from_uid(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    mu_by_validator_miner: dict[tuple[str, int], tuple[float, int]] = {}
    for key, n in cnt_by_validator_miner.items():
        s = sums_by_validator_miner.get(key, 0.0)
        mu_by_validator_miner[key] = (s / max(1, n), n)

    logger.info(
        "[weights] Element=%s Window=%s | Validator->Miner means: %s",
        element_id,
        current_window_id,
        ", ".join(f"{v}->{m}: mu={mu:.4f} (n={n})" for (v, m), (mu, n) in mu_by_validator_miner.items()),
    )

    a_rob, b_rob = 0.5, 0.5
    k = 2.5
    eps = 1e-3
    a_final, b_final = 1.0, 0.5

    miners_seen = set([m for (_v, m) in mu_by_validator_miner.keys()])
    scores_by_miner: dict[int, float] = {}

    for miner_uid in miners_seen:
        mus: list[float] = []
        wtilde: list[float] = []
        triplets: list[tuple[str, float, int]] = []

        for (validator_hk, mm), (mu, n) in mu_by_validator_miner.items():
            if mm != miner_uid:
                continue
            if n < m_min:
                continue
            stake = stake_by_hk.get(validator_hk, 0.0)
            wt = (stake**a_rob) * ((max(1, n)) ** b_rob)
            mus.append(mu)
            wtilde.append(wt)
            triplets.append((validator_hk, mu, n))

        if not mus or sum(max(0.0, w) for w in wtilde) <= 0:
            continue

        med = weighted_median(mus, wtilde)
        abs_dev = [abs(x - med) for x in mus]
        MAD = weighted_median(abs_dev, wtilde)
        if MAD < eps:
            MAD = eps
        thresh = k * (MAD / 0.6745)

        filtered: list[tuple[str, float, int]] = []
        rejected: list[tuple[str, float, int]] = []
        for validator_hk, mu, n in triplets:
            if abs(mu - med) <= thresh:
                filtered.append((validator_hk, mu, n))
            else:
                rejected.append((validator_hk, mu, n))

        if rejected:
            logger.info(
                "Element=%s Miner %d: rejected %d validator means (outliers)",
                element_id,
                miner_uid,
                len(rejected),
            )

        if len(filtered) < 2:
            VALIDATOR_MINERS_SKIPPED_TOTAL.labels(reason="insufficient_filtered").inc()
            continue

        num = 0.0
        den = 0.0
        for validator_hk, mu, n in filtered:
            stake = stake_by_hk.get(validator_hk, 0.0)
            wf = (stake**a_final) * ((max(1, n)) ** b_final)
            num += wf * mu
            den += wf
        if den <= 0:
            continue
        scores_by_miner[miner_uid] = num / den

    validator_uid = None
    try:
        validator_uid = hk_to_uid.get(get_validator_hotkey_ss58())
    except Exception:
        validator_uid = None

    if validator_uid is not None and validator_uid in scores_by_miner:
        logger.info(
            "Excluding validator uid=%d from weight candidates for element_id=%s.",
            validator_uid,
            element_id,
        )
        scores_by_miner.pop(validator_uid, None)

    if not scores_by_miner:
        logger.warning(
            "[weights] No miners passed robust filtering for element_id=%s window_id=%s.",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta_from_uid(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    VALIDATOR_MINERS_CONSIDERED.set(len(scores_by_miner))

    logger.info(
        "Element=%s Window=%s | Final miner means: %s",
        element_id,
        current_window_id,
        ", ".join(f"uid={m}: {s:.4f}" for m, s in sorted(scores_by_miner.items())),
    )

    winner_uid = max(scores_by_miner, key=scores_by_miner.get)
    logger.info(
        "Element=%s Window=%s | Provisional winner uid=%d S=%.4f over last %d blocks",
        element_id,
        current_window_id,
        winner_uid,
        scores_by_miner[winner_uid],
        tail,
    )

    if settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE:
        try:
            challenge_scores_by_validator_miner = await collect_recent_challenge_scores_by_validator_miner(
                tail=tail,
                validator_indexes=validator_indexes,
                hk_to_uid=hk_to_uid,
                element_id=element_id,
                current_window_id=current_window_id,
                K=settings.SCOREVISION_WINDOW_K_PER_VALIDATOR,
            )
            challenge_scores_by_miner = aggregate_challenge_scores_by_miner(challenge_scores_by_validator_miner)
            first_commit_block_by_hk = await _first_commit_block_by_miner(netuid)
            candidate_uids = set(scores_by_miner.keys())
            final_uid = pick_winner_with_window_tiebreak(
                winner_uid,
                uid_to_hk=uid_to_hk,
                challenge_scores_by_miner=challenge_scores_by_miner,
                candidate_uids=candidate_uids,
                delta_abs=settings.SCOREVISION_WINDOW_DELTA_ABS,
                delta_rel=settings.SCOREVISION_WINDOW_DELTA_REL,
                first_commit_block_by_hk=first_commit_block_by_hk,
                min_common_challenges=5,
            )
            if final_uid != winner_uid:
                logger.info(
                    "[window-tiebreak] Element=%s | Provisional=%d -> Final=%d",
                    element_id,
                    winner_uid,
                    final_uid,
                )
                winner_uid = final_uid
        except Exception as e:
            logger.warning("[window-tiebreak] Element=%s disabled due to error: %s", element_id, e)

    logger.info(
        "Element=%s Window=%s | Winner uid=%d (after tiebreak) over last %d blocks",
        element_id,
        current_window_id,
        winner_uid,
        tail,
    )

    TARGET_UID = 6
    final_score = float(scores_by_miner.get(winner_uid, 0.0) or 0.0)

    if abs(final_score) <= 1e-12:
        logger.info(
            "Final winner uid=%d has score=%.6f -> forcing to TARGET_UID=%d",
            winner_uid,
            final_score,
            TARGET_UID,
        )
        winner_uid = TARGET_UID
        final_score = float(scores_by_miner.get(TARGET_UID, 0.0) or 0.0)

    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(final_score)

    return winner_uid, scores_by_miner, build_winner_meta_from_uid(winner_uid, uid_to_hk, miner_meta_by_hk)


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
    r2_bucket_public_url = settings.R2_BUCKET_PUBLIC_URL

    if os.getenv("SCOREVISION_COMMIT_VALIDATOR_ON_START", "1") in ("0", "false", "False"):
        return

    try:
        index_url = None
        if r2_bucket_public_url:
            from scorevision.utils.cloudflare_helpers import build_public_index_url_from_public_base
            index_url = build_public_index_url_from_public_base(r2_bucket_public_url)
        if not index_url:
            index_url = build_public_index_url()

        if not index_url:
            logger.warning("[validator-commit] No public index URL configured; skipping.")
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
    tempo: int = 100,
    path_manifest: Path | None = None,
) -> None:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    tail_blocks_default = settings.VALIDATOR_TAIL_BLOCKS
    winners_every_n = settings.VALIDATOR_WINNERS_EVERY_N

    setup_signal_handler()
    await commit_validator_on_start(netuid)

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    subtensor = None
    last_done = -1
    effective_tail = max(tail, tail_blocks_default)
    set_weights_count = 0

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
                        if winners_by_element:
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
