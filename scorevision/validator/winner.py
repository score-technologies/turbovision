from collections import Counter, defaultdict, deque
from logging import getLogger

from scorevision.utils.bittensor_helpers import (
    _first_commit_block_by_miner,
    get_subtensor,
    get_validator_indexes_from_chain,
)
from scorevision.utils.cloudflare_helpers import dataset_sv, dataset_sv_multi
from scorevision.utils.prometheus import (
    CURRENT_WINNER,
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
)
from scorevision.utils.settings import get_settings
from scorevision.validator.models import OpenSourceMinerMeta
from scorevision.validator.payload import (
    build_winner_meta,
    extract_challenge_id,
    extract_miner_and_score,
    extract_miner_meta,
)
from scorevision.validator.scoring import (
    aggregate_challenge_scores_by_miner,
    pick_winner_with_tiebreak,
)

logger = getLogger(__name__)


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
    miner_meta_by_hk: dict[str, OpenSourceMinerMeta] = {}

    async for line in dataset_sv(tail):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            miner_meta = extract_miner_meta(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta.hotkey] = miner_meta
        except Exception:
            continue
        sums[miner_uid] = sums.get(miner_uid, 0.0) + score
        cnt[miner_uid] = cnt.get(miner_uid, 0) + 1

    if not cnt:
        logger.warning(
            "[winner] No local data for element_id=%s window_id=%s -> fallback",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    elig = [uid for uid, n in cnt.items() if n >= m_min and uid in sums]
    if not elig:
        logger.warning(
            "[winner] No miner reached %d samples for element_id=%s -> fallback",
            m_min,
            element_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    avg = {uid: (sums[uid] / cnt[uid]) for uid in elig}
    VALIDATOR_MINERS_CONSIDERED.set(len(elig))
    winner_uid = max(avg, key=avg.get)
    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(avg.get(winner_uid, 0.0))
    uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
    return winner_uid, avg, build_winner_meta(winner_uid, uid_to_hk, miner_meta_by_hk)


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
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
            challenge_id = extract_challenge_id(payload)
            if not challenge_id:
                continue
            challenge_key = f"{validator_hk}:{challenge_id}"
            challenge_scores[(validator_hk, miner_uid)].append((challenge_key, float(score)))
        except Exception:
            continue
    return challenge_scores


async def get_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    blacklisted_hotkeys: set[str] | None = None,
    validator_hotkey_ss58: str | None = None,
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

    validator_indexes = await get_validator_indexes_from_chain(netuid)
    if not validator_indexes:
        logger.warning(
            "[winner] No central validator registry found on-chain for element_id=%s",
            element_id,
        )
        return await get_local_fallback_winner_for_element(
            element_id=element_id,
            current_window_id=current_window_id,
            tail=tail,
            m_min=m_min,
            hk_to_uid=hk_to_uid,
        )

    sums_by_miner: dict[int, float] = {}
    cnt_by_miner: dict[int, int] = {}
    miner_meta_by_hk: dict[str, OpenSourceMinerMeta] = {}
    diagnostics = Counter()
    unknown_miner_hotkeys: set[str] = set()
    source_indexes: set[str] = set()

    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id):
        diagnostics["lines_total"] += 1
        src_index = str(line.get("_src_index") or "").strip()
        if src_index:
            source_indexes.add(src_index)
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                diagnostics["skip_element_mismatch"] += 1
                continue
            telemetry = payload.get("telemetry") or {}
            miner_info = telemetry.get("miner") or {}
            miner_hk = (miner_info.get("hotkey") or "").strip()
            if not miner_hk:
                diagnostics["skip_missing_miner_hotkey"] += 1
                continue
            if miner_hk not in hk_to_uid:
                diagnostics["skip_unknown_miner_hotkey"] += 1
                if len(unknown_miner_hotkeys) < 5:
                    unknown_miner_hotkeys.add(miner_hk)
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None or score is None:
                diagnostics["skip_extract_failed"] += 1
                continue
            miner_meta = extract_miner_meta(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta.hotkey] = miner_meta
        except Exception:
            diagnostics["skip_parse_error"] += 1
            continue
        diagnostics["accepted_lines"] += 1
        sums_by_miner[miner_uid] = sums_by_miner.get(miner_uid, 0.0) + score
        cnt_by_miner[miner_uid] = cnt_by_miner.get(miner_uid, 0) + 1

    if not cnt_by_miner:
        logger.warning(
            "[winner] No central validator data for element_id=%s window_id=%s -> fallback "
            "(lines=%d accepted=%d skip_unknown_hotkey=%d "
            "skip_missing_hotkey=%d skip_element=%d skip_extract=%d parse_errors=%d "
            "source_indexes=%d unknown_hotkeys=%s)",
            element_id,
            current_window_id,
            diagnostics["lines_total"],
            diagnostics["accepted_lines"],
            diagnostics["skip_unknown_miner_hotkey"],
            diagnostics["skip_missing_miner_hotkey"],
            diagnostics["skip_element_mismatch"],
            diagnostics["skip_extract_failed"],
            diagnostics["skip_parse_error"],
            len(source_indexes),
            ",".join(sorted(unknown_miner_hotkeys)) if unknown_miner_hotkeys else "-",
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    elig = [uid for uid, n in cnt_by_miner.items() if n >= m_min and uid in sums_by_miner]
    if not elig:
        logger.warning(
            "[winner] No miner reached %d samples for element_id=%s window_id=%s -> fallback",
            m_min,
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_SKIPPED_TOTAL.labels(reason="insufficient_samples").inc()
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    scores_by_miner: dict[int, float] = {
        uid: (sums_by_miner[uid] / cnt_by_miner[uid]) for uid in elig
    }

    validator_uid = None
    if validator_hotkey_ss58:
        validator_uid = hk_to_uid.get(validator_hotkey_ss58)

    if validator_uid is not None and validator_uid in scores_by_miner:
        logger.info(
            "Excluding validator uid=%d from weight candidates for element_id=%s.",
            validator_uid,
            element_id,
        )
        scores_by_miner.pop(validator_uid, None)

    if not scores_by_miner:
        logger.warning(
            "[winner] No miners eligible for element_id=%s window_id=%s.",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
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
            final_uid = pick_winner_with_tiebreak(
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

    return winner_uid, scores_by_miner, build_winner_meta(winner_uid, uid_to_hk, miner_meta_by_hk)
