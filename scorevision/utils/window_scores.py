"""
Window Score Persistence Layer
--------------------------------

Stores per-window, per-miner, per-element aggregated clip means.
Compatible with R2 shard ingestion pipeline (dataset_sv, dataset_sv_multi)
and uses the same JSONL cache structure under cloudflare_helpers.

Guarantees:
 - No double-counting shards (local + remote) by using shard_id sets.
 - Deterministic read/write order.
 - Validator can call save_window_scores() once all shards are complete.
 - EWMA integration happens at the validator layer, *not* here.

File Layout (under cache_root):
    {cache_root}/{window_id}/window_scores.jsonl
    {cache_root}/{window_id}/_seen_shards.json

JSONL row schema:
{
    "miner_id": "0x...",
    "element_id": "PlayerDetect_v1",
    "window_id": "2025-02-01",
    "clip_mean": 0.8732
}

_seen_shards.json schema:
{
    "shards": ["local:0032", "r2:0032"],
    "complete": false
}

"""

import json
from pathlib import Path
from typing import Iterable, Dict, Any
from threading import Lock
from statistics import median
from collections import defaultdict
from logging import getLogger
from functools import lru_cache
from statistics import mean
import bittensor as bt

from scorevision.utils.settings import get_settings
from scorevision.utils.prometheus import (
    CURRENT_WINNER,
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
)
from scorevision.utils.bittensor_helpers import _first_commit_block_by_miner

logger = getLogger(__name__)
# Global lock to avoid race conditions during tests / local validators
_IO_LOCK = Lock()


@lru_cache(maxsize=1)
def _validator_hotkey_ss58() -> str:
    settings = get_settings()
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return wallet.hotkey.ss58_address


# ---------------------------------------------------------------------------
# Helpers: shard tracking
# ---------------------------------------------------------------------------


def _load_seen_shards(path: Path) -> dict:
    if not path.exists():
        return {"shards": [], "complete": False}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"shards": [], "complete": False}


def _save_seen_shards(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_window_scores(cache_root: Path, window_id: str) -> list[dict[str, Any]]:
    """
    Load previously persisted window scores.

    Returns a list of JSON dicts, one per miner/element pair.
    """
    path = cache_root / window_id / "window_scores.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def save_window_scores(
    cache_root: Path,
    window_id: str,
    shard_id: str,
    rows: Iterable[Dict[str, Any]],
    mark_complete: bool = False,
) -> None:
    """
    Persist per-window aggregated clip means after a shard finishes.

    - shard_id: unique id from ingestion pipeline, e.g. "local:0032" or "r2:0032".
    - rows: iterable of { miner_id, element_id, window_id, clip_mean }.
    - mark_complete: set True when the validator determines all shards are done.

    This function appends only if shard_id has not been seen. If already seen,
    the function silently skips (idempotent writes).
    """
    window_dir = cache_root / window_id
    scores_path = window_dir / "window_scores.jsonl"
    seen_path = window_dir / "_seen_shards.json"

    with _IO_LOCK:
        seen = _load_seen_shards(seen_path)
        already = set(seen.get("shards", []))

        # Dedup: skip writes
        if shard_id in already:
            if mark_complete and not seen.get("complete"):
                seen["complete"] = True
                _save_seen_shards(seen_path, seen)
            return

        # Write rows
        window_dir.mkdir(parents=True, exist_ok=True)
        with scores_path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Update shard tracking
        already.add(shard_id)
        seen["shards"] = sorted(already)
        if mark_complete:
            seen["complete"] = True
        _save_seen_shards(seen_path, seen)


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------


def is_window_complete(cache_root: Path, window_id: str) -> bool:
    """Return True if validator has marked the window as fully ingested."""
    seen = _load_seen_shards(cache_root / window_id / "_seen_shards.json")
    return bool(seen.get("complete"))


def list_seen_shards(cache_root: Path, window_id: str) -> list[str]:
    seen = _load_seen_shards(cache_root / window_id / "_seen_shards.json")
    return list(seen.get("shards", []))


def _weighted_median(values: list[float], weights: list[float]) -> float:
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


async def compute_winner_from_window(window_summary_file: Path):
    if not window_summary_file.exists():
        return [], []

    settings = get_settings()
    stake_by_uid: dict[int, float] = {}
    S_by_miner: dict[int, float] = {}
    uid_to_hk: dict[int, str] = {}
    hk_to_uid: dict[str, int] = {}

    with window_summary_file.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
                uid = int(data["uid"])
                hk = data["hotkey"]
                mean_score = float(data["mean_score"])
                n_samples = int(data["n_samples"])
                stake = float(data.get("stake", 0.0))

                if n_samples < settings.SCOREVISION_M_MIN:
                    VALIDATOR_MINERS_SKIPPED_TOTAL.labels(
                        reason="insufficient_samples"
                    ).inc()
                    continue

                stake_by_uid[uid] = stake
                S_by_miner[uid] = mean_score
                uid_to_hk[uid] = hk
                hk_to_uid[hk] = uid
            except Exception:
                continue

    if not S_by_miner:
        return [], []

    try:
        validator_hk = _validator_hotkey_ss58()
        validator_uid = hk_to_uid.get(validator_hk)
        if validator_uid in S_by_miner:
            S_by_miner.pop(validator_uid, None)
            stake_by_uid.pop(validator_uid, None)
            logger.info(
                "Excluding validator uid=%d from candidate miners", validator_uid
            )
    except Exception:
        pass

    if not S_by_miner:
        return [], []

    VALIDATOR_MINERS_CONSIDERED.set(len(S_by_miner))

    a_final = 1.0
    final_scores: dict[int, float] = {}
    for uid, s in S_by_miner.items():
        wf = stake_by_uid.get(uid, 0.0) ** a_final
        final_scores[uid] = s * wf

    winner_uid = max(final_scores, key=final_scores.get)
    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(final_scores.get(winner_uid, 0.0))

    if settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE:
        try:
            delta_abs = settings.SCOREVISION_WINDOW_DELTA_ABS
            delta_rel = settings.SCOREVISION_WINDOW_DELTA_REL
            winner_score = final_scores[winner_uid]
            window_hi = winner_score + max(delta_abs, delta_rel * abs(winner_score))
            window_lo = winner_score - max(delta_abs, delta_rel * abs(winner_score))

            close_uids = [
                uid for uid, s in final_scores.items() if window_lo <= s <= window_hi
            ]
            if winner_uid not in close_uids:
                close_uids.append(winner_uid)

            if len(close_uids) > 1:
                first_commit_block_by_hk = await _first_commit_block_by_miner(
                    settings.SCOREVISION_NETUID
                )
                best_uid = winner_uid
                best_blk = None
                for uid in close_uids:
                    hk = uid_to_hk.get(uid)
                    blk = first_commit_block_by_hk.get(hk, 10**18)
                    if (
                        (best_blk is None)
                        or (blk < best_blk)
                        or (blk == best_blk and hk < uid_to_hk.get(best_uid, ""))
                    ):
                        best_blk = blk
                        best_uid = uid
                winner_uid = best_uid

                CURRENT_WINNER.set(winner_uid)
                VALIDATOR_WINNER_SCORE.set(final_scores.get(winner_uid, 0.0))

        except Exception as e:
            logger.warning(f"[window-tiebreak] disabled due to error: {e}")

    uids = list(S_by_miner.keys())
    scores = [S_by_miner[uid] for uid in uids]
    return uids, scores


async def aggregate_window_shards(
    cache_root: Path, tail: bool, window_id: str, min_samples: int
) -> Path:
    window_dir = cache_root / window_id
    window_dir.mkdir(parents=True, exist_ok=True)

    window_scores_file = window_dir / "window_scores.jsonl"
    summary_file = window_dir / "window_summary.jsonl"

    scores = defaultdict(list)
    if window_scores_file.exists():
        with window_scores_file.open("r") as f:
            for line in f:
                data = json.loads(line)
                scores[(data["miner_id"], data["element_id"])].append(data["clip_mean"])

    with summary_file.open("w") as f:
        for (miner_id, element_id), clip_means in scores.items():
            n_samples = len(clip_means)
            if n_samples < min_samples:
                continue
            mean_score = mean(clip_means)
            # Remove "0x" prefix if present
            uid = int(miner_id, 16) if miner_id.startswith("0x") else int(miner_id)
            f.write(
                json.dumps(
                    {
                        "uid": uid,
                        "hotkey": miner_id,
                        "mean_score": mean_score,
                        "n_samples": n_samples,
                        "stake": 0,
                    }
                )
                + "\n"
            )

    return summary_file
