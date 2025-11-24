import pytest
from json import dumps


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
    ]
    with window_file.open("w") as f:
        for l in lines:
            f.write(dumps(l) + "\n")
    return window_file
