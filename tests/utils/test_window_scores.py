import pytest
from unittest.mock import patch, AsyncMock
from json import loads, dumps

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
        uids, weights = await compute_winner_from_window(fake_window_file)

    assert uids == [2]
    assert weights == [65535]


@pytest.mark.asyncio
async def test_compute_winner_with_validator_exclusion(tmp_path):
    window_file = tmp_path / "window_scores.jsonl"
    lines = [
        {"uid": 1, "hotkey": "hk1", "mean_score": 0.9, "n_samples": 10, "stake": 1.0},
        {
            "uid": 2,
            "hotkey": "hk_validator",
            "mean_score": 1.0,
            "n_samples": 10,
            "stake": 1.0,
        },
    ]
    with window_file.open("w") as f:
        for l in lines:
            f.write(dumps(l) + "\n")

    with patch("scorevision.utils.window_scores.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.SCOREVISION_M_MIN = 1
        settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE = False

        # Exclude uid 2 as validator
        with patch(
            "scorevision.utils.window_scores._validator_hotkey_ss58",
            return_value="hk_validator",
        ):
            uids, weights = await compute_winner_from_window(window_file)

    assert uids == [1]  # validator excluded
    assert weights == [65535]


@pytest.mark.asyncio
async def test_compute_winner_with_tiebreak(tmp_path):
    window_file = tmp_path / "window_scores.jsonl"
    lines = [
        {"uid": 1, "hotkey": "hk1", "mean_score": 0.95, "n_samples": 10, "stake": 1.0},
        {"uid": 2, "hotkey": "hk2", "mean_score": 0.96, "n_samples": 10, "stake": 1.0},
    ]
    with window_file.open("w") as f:
        for l in lines:
            f.write(dumps(l) + "\n")

    # Patch settings and first_commit_block_by_miner for tie-break
    with (
        patch("scorevision.utils.window_scores.get_settings") as mock_settings,
        patch(
            "scorevision.utils.window_scores._first_commit_block_by_miner",
            new_callable=AsyncMock,
        ) as mock_first_commit,
    ):

        settings = mock_settings.return_value
        settings.SCOREVISION_M_MIN = 1
        settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE = True
        settings.SCOREVISION_WINDOW_DELTA_ABS = 0.1
        settings.SCOREVISION_WINDOW_DELTA_REL = 0.1

        mock_first_commit.return_value = {"hk1": 100, "hk2": 50}

        with patch(
            "scorevision.utils.window_scores._validator_hotkey_ss58",
            return_value="hk99",
        ):
            uids, weights = await compute_winner_from_window(window_file)

    # Tie-break should pick hk2 (earlier commit)
    assert uids == [2]
    assert weights == [65535]


@pytest.mark.asyncio
async def test_aggregate_window_shards_basic(tmp_path):
    window_id = "w1"
    window_dir = tmp_path / window_id
    window_dir.mkdir()
    rows = [
        {
            "miner_id": "0x01",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.9,
        },
        {
            "miner_id": "0x01",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.7,
        },
        {
            "miner_id": "0x02",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.8,
        },
    ]
    # Save raw shard scores
    scores_file = window_dir / "window_scores.jsonl"
    with scores_file.open("w") as f:
        for r in rows:
            f.write(dumps(r) + "\n")

    summary_file = await aggregate_window_shards(
        cache_root=tmp_path, tail=False, window_id=window_id, min_samples=1
    )
    # Check summary file content
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
