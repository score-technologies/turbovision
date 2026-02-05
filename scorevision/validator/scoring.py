from collections import deque
from statistics import median
from scorevision.utils.settings import get_settings


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


def pick_winner_with_tiebreak(
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

