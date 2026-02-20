from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import pytest
from scorevision.validator.audit.private_track.audit import (
    aggregate_scores,
    get_private_winner,
    map_hotkey_scores_to_uids,
)


def test_aggregate_scores_basic():
    results = [
        {"miner_hotkey": "hk1", "score": 0.8},
        {"miner_hotkey": "hk1", "score": 0.6},
        {"miner_hotkey": "hk2", "score": 0.9},
    ]
    scores = aggregate_scores(results)
    assert "hk1" in scores
    assert "hk2" in scores
    assert scores["hk1"] == (1.4, 2)
    assert scores["hk2"] == (0.9, 1)


def test_aggregate_scores_skips_invalid():
    results = [
        {"miner_hotkey": "hk1", "score": 0.8},
        {"miner_hotkey": None, "score": 0.5},
        {"score": 0.5},
        {"miner_hotkey": "hk2"},
    ]
    scores = aggregate_scores(results)
    assert len(scores) == 1
    assert "hk1" in scores


def test_aggregate_scores_empty():
    assert aggregate_scores([]) == {}


def test_aggregate_scores_skips_timed_out_results():
    results = [
        {"miner_hotkey": "hk1", "score": 0.9},
        {"miner_hotkey": "hk1", "score": 0.8, "timed_out": True},
        {"miner_hotkey": "hk2", "score": 0.7, "timed_out": True},
        {"miner_hotkey": "hk2", "score": 0.6},
    ]
    scores = aggregate_scores(results)
    assert scores == {
        "hk1": (0.9, 1),
        "hk2": (0.6, 1),
    }


def test_map_hotkey_scores_to_uids():
    scores_by_hotkey = {
        "hk_a": (4.0, 4),
        "hk_b": (2.0, 2),
    }
    metagraph_hotkeys = ["hk_x", "hk_a", "hk_b", "hk_y"]
    result = map_hotkey_scores_to_uids(scores_by_hotkey, metagraph_hotkeys)
    assert result == {1: (4.0, 4), 2: (2.0, 2)}


def test_map_hotkey_scores_to_uids_ignores_departed():
    scores_by_hotkey = {
        "hk_a": (4.0, 4),
        "hk_departed": (9.0, 9),
    }
    metagraph_hotkeys = ["hk_x", "hk_a", "hk_b"]
    result = map_hotkey_scores_to_uids(scores_by_hotkey, metagraph_hotkeys)
    assert result == {1: (4.0, 4)}
    assert "hk_departed" not in str(result)


_FAKE_SETTINGS = SimpleNamespace(PRIVATE_R2_PUBLIC_INDEX_URL="https://example.com/index.json")


def _patch_settings():
    return patch(
        "scorevision.validator.audit.private_track.audit.get_settings",
        return_value=_FAKE_SETTINGS,
    )


def _patch_shards(results):
    return patch(
        "scorevision.validator.audit.private_track.audit.fetch_private_shards",
        new_callable=AsyncMock,
        return_value=results,
    )


@pytest.mark.asyncio
async def test_get_private_winner():
    shards = [
        {"miner_hotkey": "hk_a", "score": 0.9},
        {"miner_hotkey": "hk_a", "score": 0.8},
        {"miner_hotkey": "hk_a", "score": 0.7},
        {"miner_hotkey": "hk_b", "score": 0.5},
        {"miner_hotkey": "hk_b", "score": 0.4},
        {"miner_hotkey": "hk_b", "score": 0.3},
    ]
    metagraph_hotkeys = ["hk_x", "hk_a", "hk_b"]

    with _patch_settings(), _patch_shards(shards):
        winner, _ = await get_private_winner(
            tail_blocks=1000,
            min_samples=2,
            metagraph_hotkeys=metagraph_hotkeys,
            blacklisted_hotkeys=set(),
        )

    assert winner == 1


@pytest.mark.asyncio
async def test_get_private_winner_no_eligible():
    shards = [
        {"miner_hotkey": "hk_a", "score": 0.9},
    ]
    metagraph_hotkeys = ["hk_x", "hk_a"]

    with _patch_settings(), _patch_shards(shards):
        winner, _ = await get_private_winner(
            tail_blocks=1000,
            min_samples=5,
            metagraph_hotkeys=metagraph_hotkeys,
            blacklisted_hotkeys=set(),
        )

    assert winner is None


@pytest.mark.asyncio
async def test_get_private_winner_blacklisted():
    shards = [
        {"miner_hotkey": "hk_a", "score": 0.9},
        {"miner_hotkey": "hk_a", "score": 0.8},
        {"miner_hotkey": "hk_b", "score": 0.5},
        {"miner_hotkey": "hk_b", "score": 0.4},
    ]
    metagraph_hotkeys = ["hk_x", "hk_a", "hk_b"]

    with _patch_settings(), _patch_shards(shards):
        winner, _ = await get_private_winner(
            tail_blocks=1000,
            min_samples=2,
            metagraph_hotkeys=metagraph_hotkeys,
            blacklisted_hotkeys={"hk_a"},
        )

    assert winner == 2
