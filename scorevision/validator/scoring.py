from collections import deque
from hashlib import sha256
from logging import getLogger
from statistics import median
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


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


def are_similar_by_challenges(
    challenge_scores1: dict[str, float],
    challenge_scores2: dict[str, float],
    *,
    delta_abs: float,
    delta_rel: float,
    min_common_challenges: int = 5,
    min_threshold_abs: float = 0.05,
    min_failed_challenges_to_reject: int = 2,
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
    failed_count = 0
    for _cid, s1, s2 in common:
        max_score = max(abs(s1), abs(s2))
        thr = max(min_threshold_abs, delta_abs, delta_rel * max_score)
        if abs(s1 - s2) > thr:
            failed_count += 1
            if failed_count >= min_failed_challenges_to_reject:
                return False
    return True


def _are_similar_by_challenges_debug(
    challenge_scores1: dict[str, float],
    challenge_scores2: dict[str, float],
    *,
    delta_abs: float,
    delta_rel: float,
    min_common_challenges: int = 5,
    min_threshold_abs: float = 0.05,
    min_failed_challenges_to_reject: int = 2,
) -> tuple[bool, dict[str, object]]:
    common_challenges = set()
    all_challenges = set(challenge_scores1.keys()) | set(challenge_scores2.keys())
    for challenge_id in all_challenges:
        score1 = float(challenge_scores1.get(challenge_id, 0.0) or 0.0)
        score2 = float(challenge_scores2.get(challenge_id, 0.0) or 0.0)
        if abs(score1) > 1e-9 and abs(score2) > 1e-9:
            common_challenges.add(challenge_id)

    debug_stats: dict[str, object] = {
        "all_challenges": len(all_challenges),
        "compared_challenges": len(common_challenges),
        "min_common_challenges": min_common_challenges,
        "failed_score_challenges": 0,
        "max_abs_diff": 0.0,
        "max_abs_diff_challenge_id": None,
        "max_abs_diff_threshold": 0.0,
        "failed_examples": [],
        "min_threshold_abs": min_threshold_abs,
        "min_failed_challenges_to_reject": min_failed_challenges_to_reject,
    }

    if len(common_challenges) < min_common_challenges:
        debug_stats["reason"] = "insufficient_common_challenges"
        return False, debug_stats

    failed_examples: list[dict[str, object]] = []
    failed_count = 0
    max_abs_diff = 0.0
    max_abs_diff_challenge_id: str | None = None
    max_abs_diff_threshold = 0.0

    for challenge_id in common_challenges:
        score1 = float(challenge_scores1.get(challenge_id, 0.0) or 0.0)
        score2 = float(challenge_scores2.get(challenge_id, 0.0) or 0.0)
        max_score = max(abs(score1), abs(score2))
        threshold = max(min_threshold_abs, delta_abs, delta_rel * max_score)
        abs_diff = abs(score1 - score2)

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_abs_diff_challenge_id = challenge_id
            max_abs_diff_threshold = threshold

        if abs_diff > threshold:
            failed_count += 1
            if len(failed_examples) < 3:
                failed_examples.append(
                    {
                        "challenge_id": challenge_id,
                        "score1": round(score1, 6),
                        "score2": round(score2, 6),
                        "abs_diff": round(abs_diff, 6),
                        "threshold": round(threshold, 6),
                    }
                )

    debug_stats["failed_score_challenges"] = failed_count
    debug_stats["max_abs_diff"] = round(max_abs_diff, 6)
    debug_stats["max_abs_diff_challenge_id"] = max_abs_diff_challenge_id
    debug_stats["max_abs_diff_threshold"] = round(max_abs_diff_threshold, 6)
    debug_stats["failed_examples"] = failed_examples

    if failed_count >= min_failed_challenges_to_reject:
        debug_stats["reason"] = "score_delta_exceeded"
        return False, debug_stats

    debug_stats["reason"] = "all_common_challenges_within_threshold"
    return True, debug_stats


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


def aggregate_challenge_score_batches_by_miner(
    challenge_scores_by_validator_miner: dict[tuple[str, int], deque],
    *,
    batch_size: int,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Return each miner's recent batch and complete evaluation-window history."""
    recent: dict[int, dict[str, float]] = {}
    history: dict[int, dict[str, float]] = {}
    size = max(1, int(batch_size))

    for (_validator_hk, miner_uid), dq in challenge_scores_by_validator_miner.items():
        rows = list(dq)
        recent_rows = rows[-size:]

        recent.setdefault(miner_uid, {})
        history.setdefault(miner_uid, {})
        for challenge_id, score in recent_rows:
            recent[miner_uid][challenge_id] = float(score)
        for challenge_id, score in rows:
            history[miner_uid][challenge_id] = float(score)

    return recent, history


def select_deterministic_historical_challenges(
    winner_history: dict[str, float],
    candidate_history: dict[str, float],
    *,
    excluded_challenge_ids: set[str],
    current_window_id: str,
    sample_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Select the same pseudo-random common history on every validator."""
    common_ids = (
        set(winner_history)
        & set(candidate_history)
    ).difference(excluded_challenge_ids)
    ranked_ids = sorted(
        common_ids,
        key=lambda challenge_id: (
            sha256(f"{current_window_id}:{challenge_id}".encode("utf-8")).digest(),
            challenge_id,
        ),
    )
    selected_ids = ranked_ids[: max(1, int(sample_size))]
    return (
        {challenge_id: winner_history[challenge_id] for challenge_id in selected_ids},
        {challenge_id: candidate_history[challenge_id] for challenge_id in selected_ids},
    )


def pick_winner_with_tiebreak(
    winner_uid: int,
    uid_to_hk: dict[int, str],
    challenge_scores_by_miner: dict[int, dict[str, float]] | None = None,
    candidate_uids: set[int] | None = None,
    *,
    recent_challenge_scores_by_miner: dict[int, dict[str, float]] | None = None,
    historical_challenge_scores_by_miner: dict[int, dict[str, float]] | None = None,
    current_window_id: str = "",
    historical_sample_size: int = 10,
    delta_abs: float,
    delta_rel: float,
    first_commit_block_by_hk: dict[str, int],
    min_common_challenges: int = 5,
) -> int:
    legacy_single_batch = recent_challenge_scores_by_miner is None
    if recent_challenge_scores_by_miner is None:
        recent_challenge_scores_by_miner = challenge_scores_by_miner or {}
    if historical_challenge_scores_by_miner is None:
        historical_challenge_scores_by_miner = (
            recent_challenge_scores_by_miner if legacy_single_batch else {}
        )
    candidate_uids = set(candidate_uids or set())

    if winner_uid not in recent_challenge_scores_by_miner:
        return winner_uid
    winner_recent_scores = recent_challenge_scores_by_miner[winner_uid]
    similar_uids = [winner_uid]
    for miner_uid in candidate_uids:
        if miner_uid == winner_uid:
            continue
        recent_scores = recent_challenge_scores_by_miner.get(miner_uid)
        if not recent_scores:
            continue
        recent_is_similar, recent_debug = _are_similar_by_challenges_debug(
            winner_recent_scores,
            recent_scores,
            delta_abs=delta_abs,
            delta_rel=delta_rel,
            min_common_challenges=min_common_challenges,
        )

        logger.info(
            "[window-tiebreak] recent-batch winner uid=%d hk=%s vs uid=%d hk=%s -> "
            "similar=%s reason=%s all=%s compared=%s min_common=%s failed_score=%s "
            "max_abs_diff=%s max_abs_diff_threshold=%s max_diff_challenge=%s "
            "failed_examples=%s",
            winner_uid,
            uid_to_hk.get(winner_uid, ""),
            miner_uid,
            uid_to_hk.get(miner_uid, ""),
            recent_is_similar,
            recent_debug.get("reason"),
            recent_debug.get("all_challenges"),
            recent_debug.get("compared_challenges"),
            recent_debug.get("min_common_challenges"),
            recent_debug.get("failed_score_challenges"),
            recent_debug.get("max_abs_diff"),
            recent_debug.get("max_abs_diff_threshold"),
            recent_debug.get("max_abs_diff_challenge_id"),
            recent_debug.get("failed_examples"),
        )
        if not recent_is_similar:
            continue
        if legacy_single_batch:
            similar_uids.append(miner_uid)
            continue

        winner_history = historical_challenge_scores_by_miner.get(winner_uid)
        candidate_history = historical_challenge_scores_by_miner.get(miner_uid)
        if not winner_history or not candidate_history:
            continue
        excluded_recent_ids = set(winner_recent_scores) | set(recent_scores)
        winner_historical_sample, candidate_historical_sample = (
            select_deterministic_historical_challenges(
                winner_history,
                candidate_history,
                excluded_challenge_ids=excluded_recent_ids,
                current_window_id=current_window_id,
                sample_size=historical_sample_size,
            )
        )
        historical_is_similar, historical_debug = _are_similar_by_challenges_debug(
            winner_historical_sample,
            candidate_historical_sample,
            delta_abs=delta_abs,
            delta_rel=delta_rel,
            min_common_challenges=min_common_challenges,
        )
        logger.info(
            "[window-tiebreak] historical-sample winner uid=%d hk=%s vs uid=%d hk=%s -> "
            "similar=%s reason=%s all=%s compared=%s min_common=%s failed_score=%s "
            "max_abs_diff=%s max_abs_diff_threshold=%s max_diff_challenge=%s "
            "failed_examples=%s",
            winner_uid,
            uid_to_hk.get(winner_uid, ""),
            miner_uid,
            uid_to_hk.get(miner_uid, ""),
            historical_is_similar,
            historical_debug.get("reason"),
            historical_debug.get("all_challenges"),
            historical_debug.get("compared_challenges"),
            historical_debug.get("min_common_challenges"),
            historical_debug.get("failed_score_challenges"),
            historical_debug.get("max_abs_diff"),
            historical_debug.get("max_abs_diff_threshold"),
            historical_debug.get("max_abs_diff_challenge_id"),
            historical_debug.get("failed_examples"),
        )
        if historical_is_similar:
            similar_uids.append(miner_uid)
    if len(similar_uids) == 1:
        logger.info("[window-tiebreak] No similar miners found; provisional winner %d wins", winner_uid)
        return winner_uid
    logger.info("[window-tiebreak] Found %d similar miners: %s", len(similar_uids), similar_uids)
    best_uid = winner_uid
    best_blk = None
    for miner_uid in similar_uids:
        hk = uid_to_hk.get(miner_uid)
        if not hk:
            continue
        blk = first_commit_block_by_hk.get(hk)
        candidate = int(blk) if blk is not None else 10**18
        if (
            (best_blk is None)
            or (candidate < best_blk)
            or (candidate == best_blk and hk < uid_to_hk.get(best_uid, ""))
        ):
            best_blk = candidate
            best_uid = miner_uid
    return best_uid
