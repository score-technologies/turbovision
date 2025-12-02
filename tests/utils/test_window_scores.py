import pytest
from unittest.mock import patch, AsyncMock
from json import loads

from scorevision.utils.window_scores import (
    save_window_scores,
    load_window_scores,
    aggregate_window_shards,
    compute_winner_from_window,
    is_window_complete,
    list_seen_shards,
)


def test_save_and_load_window_scores(cache_root):
    window_id = "window-001"
    shard_id = "local:0001"
    rows = [
        {
            "miner_id": "0x01",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.9,
        },
        {
            "miner_id": "0x02",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.8,
        },
    ]

    # Save scores
    save_window_scores(cache_root, window_id, shard_id, rows)
    loaded = load_window_scores(cache_root, window_id)
    assert len(loaded) == 2
    assert loaded[0]["clip_mean"] == 0.9

    # Check seen shards
    shards = list_seen_shards(cache_root, window_id)
    assert shard_id in shards
    assert not is_window_complete(cache_root, window_id)

    # Mark complete
    save_window_scores(cache_root, window_id, "local:0002", [], mark_complete=True)
    assert is_window_complete(cache_root, window_id)


def test_dedup_shard_write(cache_root):
    window_id = "window-002"
    shard_id = "local:0001"
    rows = [
        {
            "miner_id": "0x01",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.7,
        }
    ]

    save_window_scores(cache_root, window_id, shard_id, rows)
    # Writing same shard again should not duplicate
    save_window_scores(cache_root, window_id, shard_id, [{"clip_mean": 0.99}])
    loaded = load_window_scores(cache_root, window_id)
    assert len(loaded) == 1
    assert loaded[0]["clip_mean"] == 0.7  # original


@pytest.mark.asyncio
async def test_compute_winner_basic(fake_window_file, fake_settings):
    with (
        patch(
            "scorevision.utils.window_scores.get_settings", return_value=fake_settings
        ),
        patch(
            "scorevision.utils.window_scores._validator_hotkey_ss58",
            return_value="hk99",
        ),
    ):
        uids, weights, winner_uid = await compute_winner_from_window(fake_window_file)

    # On ne suppose plus que uids == [winner]
    assert uids
    assert len(uids) == len(weights)

    # On teste juste le gagnant attendu (ancien comportement : uid 4)
    assert winner_uid == 4


@pytest.mark.asyncio
async def test_compute_winner_with_validator_exclusion(fake_window_file, fake_settings):
    with (
        patch(
            "scorevision.utils.window_scores.get_settings", return_value=fake_settings
        ),
        patch(
            "scorevision.utils.window_scores._validator_hotkey_ss58",
            return_value="hk_validator",
        ),
    ):
        uids, weights, winner_uid = await compute_winner_from_window(fake_window_file)

    assert uids
    assert len(uids) == len(weights)

    # Gagnant attendu après exclusion du validator = uid 2
    assert winner_uid == 2


@pytest.mark.asyncio
async def test_compute_winner_with_tiebreak(fake_window_file, fake_settings):
    fake_settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE = True
    with (
        patch(
            "scorevision.utils.window_scores.get_settings", return_value=fake_settings
        ),
        patch(
            "scorevision.utils.window_scores._validator_hotkey_ss58",
            return_value="hk99",
        ),
        patch(
            "scorevision.utils.window_scores._first_commit_block_by_miner",
            return_value=AsyncMock,
        ) as mock_first_commit,
    ):
        # hk1 & hk2 sont très proches en score → tiebreak sur block
        mock_first_commit.return_value = {"hk1": 100, "hk2": 50}
        uids, weights, winner_uid = await compute_winner_from_window(fake_window_file)

    assert uids
    assert len(uids) == len(weights)

    # Le tiebreak doit favoriser hk2 → uid 2
    assert winner_uid == 2


@pytest.mark.asyncio
async def test_aggregate_window_shards_basic(fake_window_shards_path):
    summary_file = await aggregate_window_shards(
        cache_root=fake_window_shards_path, tail=False, window_id="w1", min_samples=1
    )
    with summary_file.open("r") as f:
        summary_lines = [loads(l) for l in f]

    assert any(
        s["hotkey"] == "0x01" and abs(s["mean_score"] - 0.8) < 1e-6
        for s in summary_lines
    )
    assert any(
        s["hotkey"] == "0x02" and abs(s["mean_score"] - 0.8) < 1e-6
        for s in summary_lines
    )