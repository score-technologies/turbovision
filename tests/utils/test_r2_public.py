from scorevision.utils.r2_public import (
    extract_element_miner_commit_from_key,
    filter_keys_by_latest_commit_by_miner,
)


def test_extract_element_miner_commit_from_key_with_commit_folder():
    key = "manako/elemA/hk1/000000123/evaluation/000000999-challenge.json"
    assert extract_element_miner_commit_from_key(key) == ("elemA", "hk1", 123)


def test_extract_element_miner_commit_from_key_legacy_layout():
    key = "manako/elemA/hk1/evaluation/000000999-challenge.json"
    assert extract_element_miner_commit_from_key(key) == ("elemA", "hk1", -1)


def test_filter_keys_by_latest_commit_by_miner_keeps_only_latest():
    keys = [
        "manako/elemA/hk1/000000123/evaluation/000000900-c1.json",
        "manako/elemA/hk1/000000124/evaluation/000000901-c2.json",
        "manako/elemA/hk2/000000050/evaluation/000000800-c3.json",
    ]
    filtered = filter_keys_by_latest_commit_by_miner(keys)
    assert "manako/elemA/hk1/000000123/evaluation/000000900-c1.json" not in filtered
    assert "manako/elemA/hk1/000000124/evaluation/000000901-c2.json" in filtered
    assert "manako/elemA/hk2/000000050/evaluation/000000800-c3.json" in filtered
