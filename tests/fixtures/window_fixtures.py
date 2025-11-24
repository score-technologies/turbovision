import pytest
from json import dumps
from pathlib import Path


@pytest.fixture
def fake_window_file(tmp_path):
    """
    Creates a JSONL file with sample window scores for 3 miners.
    Returns the Path to the file.
    """
    window_file = tmp_path / "window_scores.jsonl"
    lines = [
        {"uid": 1, "hotkey": "hk1", "mean_score": 0.8, "n_samples": 10, "stake": 1.0},
        {"uid": 2, "hotkey": "hk2", "mean_score": 0.9, "n_samples": 10, "stake": 1.0},
        {"uid": 3, "hotkey": "hk3", "mean_score": 0.85, "n_samples": 10, "stake": 0.5},
        {
            "uid": 4,
            "hotkey": "hk_validator",
            "mean_score": 1.0,
            "n_samples": 10,
            "stake": 1.0,
        },
    ]
    with window_file.open("w") as f:
        for l in lines:
            f.write(dumps(l) + "\n")
    return window_file


@pytest.fixture
def fake_window_shards_path(tmp_path) -> Path:
    """
    Creates a JSONL file with raw shard scores for multiple miners/elements.
    Returns the Path to the file, suitable for testing `aggregate_window_shards`.
    """
    window_id = "w1"
    window_dir = tmp_path / window_id
    window_dir.mkdir(parents=True, exist_ok=True)

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
        {
            "miner_id": "0x02",
            "element_id": "E2",
            "window_id": window_id,
            "clip_mean": 0.6,
        },
        {
            "miner_id": "0x03",
            "element_id": "E1",
            "window_id": window_id,
            "clip_mean": 0.85,
        },
    ]

    scores_file = window_dir / "window_scores.jsonl"
    with scores_file.open("w") as f:
        for row in rows:
            f.write(dumps(row) + "\n")

    return tmp_path
