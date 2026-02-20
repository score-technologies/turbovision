import logging
from collections import defaultdict
from scorevision.utils.r2_public import (
    fetch_index_keys,
    fetch_shard_lines,
    filter_keys_by_tail,
)
from scorevision.utils.settings import get_settings

logger = logging.getLogger(__name__)


async def fetch_private_shards(
    public_index_url: str, tail_blocks: int
) -> list[dict]:
    index_keys = await fetch_index_keys(public_index_url)
    if not index_keys:
        logger.warning("[private-audit] No index keys found")
        return []

    filtered_keys, max_block, min_keep = filter_keys_by_tail(
        index_keys, tail_blocks
    )
    logger.info(
        "[private-audit] Fetched %d keys, %d within tail (blocks %d-%d)",
        len(index_keys),
        len(filtered_keys),
        min_keep,
        max_block,
    )

    all_results: list[dict] = []
    for key in filtered_keys:
        lines = await fetch_shard_lines(public_index_url, key)
        all_results.extend(lines)

    return all_results


def aggregate_scores(results: list[dict]) -> dict[str, tuple[float, int]]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for result in results:
        hotkey = result.get("miner_hotkey")
        score = result.get("score")
        if not hotkey or score is None:
            continue
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        sums[hotkey] += score
        counts[hotkey] += 1

    return {hk: (sums[hk], counts[hk]) for hk in sums}


def map_hotkey_scores_to_uids(
    scores_by_hotkey: dict[str, tuple[float, int]],
    metagraph_hotkeys: list[str],
) -> dict[int, tuple[float, int]]:
    hotkey_to_uid = {hk: uid for uid, hk in enumerate(metagraph_hotkeys)}
    scores_by_uid: dict[int, tuple[float, int]] = {}
    for hk, value in scores_by_hotkey.items():
        uid = hotkey_to_uid.get(hk)
        if uid is not None:
            scores_by_uid[uid] = value
    return scores_by_uid


async def get_private_winner(
    tail_blocks: int,
    min_samples: int,
    metagraph_hotkeys: list[str],
    blacklisted_hotkeys: set[str],
) -> tuple[int | None, dict[str, str | None] | None]:
    settings = get_settings()
    public_index_url = settings.PRIVATE_R2_PUBLIC_INDEX_URL
    if not public_index_url:
        logger.warning("[private-audit] PRIVATE_R2_PUBLIC_INDEX_URL not configured")
        return None, None

    results = await fetch_private_shards(public_index_url, tail_blocks)
    if not results:
        logger.warning("[private-audit] No evaluation results found")
        return None, None

    scores_by_hotkey = aggregate_scores(results)
    if not scores_by_hotkey:
        logger.warning("[private-audit] No valid scores to aggregate")
        return None, None

    for hk in list(scores_by_hotkey.keys()):
        if hk in blacklisted_hotkeys:
            logger.info("[private-audit] Removing blacklisted hotkey=%s", hk)
            scores_by_hotkey.pop(hk)

    scores_by_uid = map_hotkey_scores_to_uids(scores_by_hotkey, metagraph_hotkeys)
    if not scores_by_uid:
        logger.warning("[private-audit] No active UIDs match aggregated hotkeys")
        return None, None

    avg_by_uid: dict[int, float] = {}
    for uid, (total, count) in scores_by_uid.items():
        if count >= min_samples:
            avg_by_uid[uid] = total / count

    if not avg_by_uid:
        logger.warning("[private-audit] No miners reached %d samples", min_samples)
        return None, None

    winner_uid = max(avg_by_uid, key=avg_by_uid.get)
    winner_hotkey = (
        metagraph_hotkeys[winner_uid]
        if 0 <= winner_uid < len(metagraph_hotkeys)
        else None
    )
    winner_meta = (
        {
            "hotkey": winner_hotkey,
            "chute_id": None,
            "slug": None,
        }
        if winner_hotkey
        else None
    )
    logger.info(
        "[private-audit] Winner uid=%d score=%.4f | %s",
        winner_uid,
        avg_by_uid[winner_uid],
        ", ".join(f"uid={u}: {s:.4f}" for u, s in sorted(avg_by_uid.items())),
    )
    return winner_uid, winner_meta
