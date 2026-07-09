from collections import Counter, defaultdict, deque
from logging import getLogger
import math

from scorevision.utils.bittensor_helpers import (
    _first_commit_block_by_miner,
    get_subtensor,
    get_validator_indexes_from_chain,
)
from scorevision.utils.cloudflare_helpers import dataset_sv, dataset_sv_multi
from scorevision.utils.compliance_failures import (
    ComplianceFailureTuple,
    is_compliance_tuple_failed,
)
from scorevision.utils.prometheus import (
    CURRENT_WINNER,
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
)
from scorevision.utils.settings import get_settings
from scorevision.utils.r2_public import extract_element_miner_commit_from_key
from scorevision.validator.models import OpenSourceMinerMeta
from scorevision.validator.payload import (
    build_winner_meta,
    extract_challenge_id,
    extract_miner_and_score,
    extract_miner_meta,
)
from scorevision.validator.scoring import (
    aggregate_challenge_scores_by_miner,
    days_to_blocks,
    pick_winner_with_tiebreak,
)

logger = getLogger(__name__)


def _drop_initial_zero_scores(scores: list[float], *, max_dropped: int = 5) -> tuple[list[float], int]:
    dropped = 0
    for score in scores:
        if dropped >= max(0, int(max_dropped)):
            break
        if abs(float(score or 0.0)) <= 1e-12:
            dropped += 1
            continue
        break
    if dropped <= 0:
        return list(scores), 0
    return list(scores[dropped:]), dropped


def _extract_sample_block(line: dict, payload: dict, telemetry: dict) -> int | None:
    for block_value in (payload.get("block"), telemetry.get("block"), line.get("block")):
        if block_value is None:
            continue
        try:
            return int(block_value)
        except Exception:
            continue
    return None


def _extract_sample_commit_block(line: dict) -> int | None:
    for key_name in ("_key", "key", "path", "url"):
        key = line.get(key_name)
        if not key:
            continue
        try:
            _element, _miner, commit_block = extract_element_miner_commit_from_key(str(key))
        except Exception:
            continue
        if commit_block >= 0:
            return int(commit_block)
    return None


def _apply_recent_commit_initial_zero_filter(
    *,
    samples_by_uid: dict[int, list[tuple[int, float]]],
    uid_to_hk: dict[int, str],
    first_commit_block_by_hk: dict[str, int],
    max_block: int | None,
    recent_commit_blocks: int,
    enabled: bool = True,
) -> tuple[dict[int, list[float]], dict[int, int]]:
    filtered_scores_by_uid: dict[int, list[float]] = {}
    dropped_zero_prefix_by_uid: dict[int, int] = {}

    for uid, samples in samples_by_uid.items():
        ordered_samples = sorted(samples, key=lambda sample: int(sample[0]))
        ordered_scores = [float(score) for (_block, score) in ordered_samples]
        hk = uid_to_hk.get(uid, "")
        commit_block = first_commit_block_by_hk.get(hk) if hk else None

        should_filter = (
            enabled
            and max_block is not None
            and commit_block is not None
            and int(max_block) - int(commit_block) <= int(recent_commit_blocks)
        )

        if not should_filter or not ordered_scores:
            filtered_scores_by_uid[uid] = ordered_scores
            dropped_zero_prefix_by_uid[uid] = 0
            continue

        filtered_scores, dropped = _drop_initial_zero_scores(ordered_scores)
        filtered_scores_by_uid[uid] = filtered_scores
        dropped_zero_prefix_by_uid[uid] = dropped

    return filtered_scores_by_uid, dropped_zero_prefix_by_uid


def compute_adaptive_delta_rel(
    *,
    default_delta_rel: float,
    baseline_theta: float | None,
    min_delta_rel: float = 0.04,
    max_delta_rel: float = 0.175,
) -> float:
    if baseline_theta is None:
        return float(default_delta_rel)
    try:
        base_model_score = float(baseline_theta)
    except Exception:
        return float(default_delta_rel)
    if not math.isfinite(base_model_score):
        return float(default_delta_rel)
    adaptive_delta_rel = min_delta_rel + (max_delta_rel - min_delta_rel) * base_model_score
    return max(min_delta_rel, min(max_delta_rel, adaptive_delta_rel))


async def get_local_fallback_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    hk_to_uid: dict[str, int],
    lane: str = "public",
    compliance_failure_tuples: set[ComplianceFailureTuple] | None = None,
) -> tuple[int | None, dict[int, float], dict[str, str | None] | None]:
    settings = get_settings()
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    sums: dict[int, float] = {}
    cnt: dict[int, int] = {}
    miner_meta_by_hk: dict[str, OpenSourceMinerMeta] = {}

    async for line in dataset_sv(tail, lane=lane):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_lane = str(payload.get("lane") or "public").strip() or "public"
            if lane and payload_lane != lane:
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            telemetry = payload.get("telemetry") or {}
            miner_info = telemetry.get("miner") or {}
            miner_hk = (miner_info.get("hotkey") or "").strip()
            commit_block = _extract_sample_commit_block(line)
            if is_compliance_tuple_failed(
                compliance_failure_tuples,
                hotkey=miner_hk,
                element_id=element_id,
                commit_block=commit_block,
            ):
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
    lane: str = "public",
    K: int = 25,
    eligible_uids: set[int] | None = None,
    excluded_uids: set[int] | None = None,
    compliance_failure_tuples: set[ComplianceFailureTuple] | None = None,
) -> dict[tuple[str, int], deque]:
    challenge_scores: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=K))
    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id, lane=lane):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            telemetry = payload.get("telemetry") or {}
            miner_info = telemetry.get("miner") or {}
            miner_hk = (miner_info.get("hotkey") or "").strip()
            commit_block = _extract_sample_commit_block(line)
            if is_compliance_tuple_failed(
                compliance_failure_tuples,
                hotkey=miner_hk,
                element_id=element_id,
                commit_block=commit_block,
            ):
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            if eligible_uids is not None and miner_uid not in eligible_uids:
                continue
            if excluded_uids is not None and miner_uid in excluded_uids:
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
    baseline_theta: float | None = None,
    first_block: int | None = None,
    blacklisted_hotkeys: set[str] | None = None,
    validator_hotkey_ss58: str | None = None,
    lane: str = "public",
    compliance_failure_tuples: set[ComplianceFailureTuple] | None = None,
) -> tuple[
    int | None,
    dict[int, float],
    dict[str, str | None] | None,
    list[dict[str, float | int | str]],
]:
    settings = get_settings()
    subtensor = await get_subtensor()
    netuid = settings.SCOREVISION_NETUID
    mechid = settings.SCOREVISION_MECHID
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    normalized_lane = str(lane or "public").strip() or "public"

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
        winner_uid, scores_by_uid, winner_meta = await get_local_fallback_winner_for_element(
            element_id=element_id,
            current_window_id=current_window_id,
            tail=tail,
            m_min=m_min,
            hk_to_uid=hk_to_uid,
            lane=lane,
            compliance_failure_tuples=compliance_failure_tuples,
        )
        return winner_uid, scores_by_uid, winner_meta, []

    sums_by_miner: dict[int, float] = {}
    cnt_by_miner: dict[int, int] = {}
    samples_by_miner: dict[int, list[tuple[int, float]]] = defaultdict(list)
    miner_meta_by_hk: dict[str, OpenSourceMinerMeta] = {}
    diagnostics = Counter()
    unknown_miner_hotkeys: set[str] = set()
    compliance_failed_hotkeys: set[str] = set()
    source_indexes: set[str] = set()
    max_observed_block: int | None = None

    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id, lane=lane):
        diagnostics["lines_total"] += 1
        src_index = str(line.get("_src_index") or "").strip()
        if src_index:
            source_indexes.add(src_index)
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                diagnostics["skip_element_mismatch"] += 1
                continue
            payload_lane = str(payload.get("lane") or "public").strip() or "public"
            if lane and payload_lane != lane:
                diagnostics["skip_lane_mismatch"] += 1
                continue
            telemetry = payload.get("telemetry") or {}
            miner_info = telemetry.get("miner") or {}
            miner_hk = (miner_info.get("hotkey") or "").strip()
            if not miner_hk:
                diagnostics["skip_missing_miner_hotkey"] += 1
                continue
            commit_block = _extract_sample_commit_block(line)
            if is_compliance_tuple_failed(
                compliance_failure_tuples,
                hotkey=miner_hk,
                element_id=element_id,
                commit_block=commit_block,
            ):
                diagnostics["skip_compliance_failed_tuple"] += 1
                compliance_failed_hotkeys.add(miner_hk)
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
            block_int = _extract_sample_block(line, payload, telemetry)
        except Exception:
            diagnostics["skip_parse_error"] += 1
            continue
        diagnostics["accepted_lines"] += 1
        samples_by_miner[miner_uid].append((block_int or 0, float(score)))
        if block_int is not None and (max_observed_block is None or block_int > max_observed_block):
            max_observed_block = block_int

    if compliance_failure_tuples or diagnostics["skip_compliance_failed_tuple"]:
        logger.info(
            "[winner:compliance] element_id=%s window_id=%s loaded_failing_tuples=%d "
            "skipped_samples=%d skipped_hotkeys=%d",
            element_id,
            current_window_id,
            len(compliance_failure_tuples or ()),
            diagnostics["skip_compliance_failed_tuple"],
            len(compliance_failed_hotkeys),
        )

    first_commit_block_by_hk = await _first_commit_block_by_miner(
        netuid,
        element_id=element_id,
        candidate_hotkeys={uid_to_hk[uid] for uid in samples_by_miner.keys() if uid in uid_to_hk},
        first_block=first_block,
    )
    recent_commit_blocks = days_to_blocks(3) or 0
    initial_zero_filter_enabled = normalized_lane != "private"
    filtered_scores_by_uid, dropped_zero_prefix_by_uid = _apply_recent_commit_initial_zero_filter(
        samples_by_uid=samples_by_miner,
        uid_to_hk=uid_to_hk,
        first_commit_block_by_hk=first_commit_block_by_hk,
        max_block=max_observed_block,
        recent_commit_blocks=recent_commit_blocks,
        enabled=initial_zero_filter_enabled,
    )

    dropped_total = 0
    for uid, scores in filtered_scores_by_uid.items():
        dropped = int(dropped_zero_prefix_by_uid.get(uid, 0))
        dropped_total += dropped
        if not scores:
            continue
        sums_by_miner[uid] = sum(scores)
        cnt_by_miner[uid] = len(scores)
    if dropped_total > 0:
        logger.info(
            "[weights:warmup-filter] element_id=%s dropped_initial_zero_scores=%d miners_affected=%d recent_window_blocks=%d max_observed_block=%s",
            element_id,
            dropped_total,
            sum(1 for v in dropped_zero_prefix_by_uid.values() if int(v) > 0),
            recent_commit_blocks,
            max_observed_block,
        )
    elif not initial_zero_filter_enabled:
        logger.info("[weights:warmup-filter] element_id=%s disabled for private lane", element_id)

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
            [],
        )

    validator_uid = None
    if validator_hotkey_ss58:
        validator_uid = hk_to_uid.get(validator_hotkey_ss58)

    sample_rows_all: list[dict[str, float | int | str]] = []
    for uid, n in cnt_by_miner.items():
        if uid not in sums_by_miner:
            continue
        if validator_uid is not None and uid == validator_uid:
            continue
        hk = uid_to_hk.get(uid)
        if not hk:
            continue
        sample_rows_all.append(
            {
                "hotkey": hk,
                "uid": int(uid),
                "avg_score": float(sums_by_miner[uid] / n),
                "n_challenges": int(n),
            }
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
            sample_rows_all,
        )

    scores_by_miner: dict[int, float] = {
        uid: (sums_by_miner[uid] / cnt_by_miner[uid]) for uid in elig
    }

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
            sample_rows_all,
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

    winner_from_tiebreak_only_pool = False
    tiebreak_enabled_for_lane = settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE and normalized_lane != "private"
    if tiebreak_enabled_for_lane:
        try:
            adaptive_delta_rel = compute_adaptive_delta_rel(
                default_delta_rel=settings.SCOREVISION_WINDOW_DELTA_REL,
                baseline_theta=baseline_theta,
            )
            logger.info(
                "[window-tiebreak] Element=%s | baseline_theta=%s -> delta_rel=%.6f (default=%.6f)",
                element_id,
                baseline_theta,
                adaptive_delta_rel,
                settings.SCOREVISION_WINDOW_DELTA_REL,
            )
            zero_score_eps = 1e-12
            eligible_non_zero_uids = {
                uid for uid, score in scores_by_miner.items() if abs(float(score or 0.0)) > zero_score_eps
            }
            excluded_zero_uids = [
                uid
                for uid, score in scores_by_miner.items()
                if uid != winner_uid and abs(float(score or 0.0)) <= zero_score_eps
            ]
            candidate_uids = set(eligible_non_zero_uids)
            candidate_uids.add(winner_uid)
            if excluded_zero_uids:
                logger.info(
                    "[window-tiebreak] Element=%s | Excluding %d zero-score miners from candidate set",
                    element_id,
                    len(excluded_zero_uids),
                )

            excluded_uids_for_tiebreak: set[int] = set()
            if validator_uid is not None:
                excluded_uids_for_tiebreak.add(validator_uid)

            challenge_scores_by_validator_miner = await collect_recent_challenge_scores_by_validator_miner(
                tail=tail,
                validator_indexes=validator_indexes,
                hk_to_uid=hk_to_uid,
                element_id=element_id,
                current_window_id=current_window_id,
                lane=lane,
                K=settings.SCOREVISION_WINDOW_K_PER_VALIDATOR,
                eligible_uids=None,
                excluded_uids=excluded_uids_for_tiebreak,
                compliance_failure_tuples=compliance_failure_tuples,
            )
            challenge_scores_by_miner = aggregate_challenge_scores_by_miner(challenge_scores_by_validator_miner)

            additional_tiebreak_uids = {
                uid for uid in challenge_scores_by_miner.keys() if uid not in excluded_uids_for_tiebreak
            }
            if additional_tiebreak_uids:
                candidate_uids.update(additional_tiebreak_uids)

            tiebreak_only_uids = candidate_uids.difference(set(scores_by_miner.keys()))
            if tiebreak_only_uids:
                logger.info(
                    "[window-tiebreak] Element=%s | Added %d non-eligible miners to tiebreak pool",
                    element_id,
                    len(tiebreak_only_uids),
                )

            first_commit_block_by_hk_tiebreak = await _first_commit_block_by_miner(
                netuid,
                element_id=element_id,
                candidate_hotkeys={uid_to_hk[uid] for uid in candidate_uids if uid in uid_to_hk},
                backfill_allowed_hotkeys={
                    uid_to_hk[uid]
                    for uid in challenge_scores_by_miner.keys()
                    if uid in uid_to_hk
                },
                first_block=first_block,
            )
            final_uid = pick_winner_with_tiebreak(
                winner_uid,
                uid_to_hk=uid_to_hk,
                challenge_scores_by_miner=challenge_scores_by_miner,
                candidate_uids=candidate_uids,
                delta_abs=settings.SCOREVISION_WINDOW_DELTA_ABS,
                delta_rel=adaptive_delta_rel,
                first_commit_block_by_hk=first_commit_block_by_hk_tiebreak,
                min_common_challenges=6,
            )
            if final_uid != winner_uid:
                logger.info(
                    "[window-tiebreak] Element=%s | Provisional=%d -> Final=%d",
                    element_id,
                    winner_uid,
                    final_uid,
                )
                winner_uid = final_uid
            winner_from_tiebreak_only_pool = winner_uid in tiebreak_only_uids
        except Exception as e:
            logger.warning("[window-tiebreak] Element=%s disabled due to error: %s", element_id, e)
    elif normalized_lane == "private" and settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE:
        logger.info("[window-tiebreak] Element=%s disabled for private lane", element_id)

    logger.info(
        "Element=%s Window=%s | Winner uid=%d (after tiebreak) over last %d blocks",
        element_id,
        current_window_id,
        winner_uid,
        tail,
    )

    TARGET_UID = 6
    winner_has_eligible_score = winner_uid in scores_by_miner
    final_score = float(scores_by_miner.get(winner_uid, 0.0) or 0.0)

    if winner_from_tiebreak_only_pool or not winner_has_eligible_score:
        logger.info(
            "[window-tiebreak] Element=%s | Winner uid=%d came from non-eligible pool -> routing to TARGET_UID=%d",
            element_id,
            winner_uid,
            TARGET_UID,
        )
        winner_uid = TARGET_UID
        final_score = float(scores_by_miner.get(TARGET_UID, 0.0) or 0.0)

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

    return winner_uid, scores_by_miner, build_winner_meta(winner_uid, uid_to_hk, miner_meta_by_hk), sample_rows_all
