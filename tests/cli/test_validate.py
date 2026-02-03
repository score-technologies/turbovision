import pytest
from types import SimpleNamespace
from scorevision.cli.validate import (
    _extract_miner_and_score_from_payload,
    _extract_miner_meta_from_payload,
    _build_winner_meta_from_uid,
    _weighted_median,
    _days_to_blocks,
    _stake_of,
    _extract_challenge_id_from_payload,
    _are_similar_by_challenges,
    _extract_elements_from_manifest,
)


def test_extract_miner_and_score_from_payload_valid():
    payload = {
        "telemetry": {"miner": {"hotkey": "hk123"}},
        "metrics": {"composite_score": 0.85},
    }
    hk_to_uid = {"hk123": 5}
    uid, score = _extract_miner_and_score_from_payload(payload, hk_to_uid)
    assert uid == 5
    assert score == 0.85


def test_extract_miner_and_score_from_payload_score_in_root():
    payload = {
        "telemetry": {"miner": {"hotkey": "hk456"}},
        "composite_score": 0.75,
    }
    hk_to_uid = {"hk456": 10}
    uid, score = _extract_miner_and_score_from_payload(payload, hk_to_uid)
    assert uid == 10
    assert score == 0.75


def test_extract_miner_and_score_from_payload_unknown_hotkey():
    payload = {
        "telemetry": {"miner": {"hotkey": "unknown_hk"}},
        "metrics": {"composite_score": 0.5},
    }
    hk_to_uid = {"hk123": 5}
    uid, score = _extract_miner_and_score_from_payload(payload, hk_to_uid)
    assert uid is None
    assert score is None


def test_extract_miner_and_score_from_payload_empty_hotkey():
    payload = {
        "telemetry": {"miner": {"hotkey": ""}},
        "metrics": {"composite_score": 0.5},
    }
    hk_to_uid = {"hk123": 5}
    uid, score = _extract_miner_and_score_from_payload(payload, hk_to_uid)
    assert uid is None
    assert score is None


def test_extract_miner_meta_from_payload_valid():
    payload = {
        "telemetry": {
            "miner": {
                "hotkey": "hk123",
                "chute_id": "chute_abc",
                "slug": "my-miner",
            }
        }
    }
    meta = _extract_miner_meta_from_payload(payload)
    assert meta == {
        "hotkey": "hk123",
        "chute_id": "chute_abc",
        "slug": "my-miner",
    }


def test_extract_miner_meta_from_payload_empty_hotkey():
    payload = {"telemetry": {"miner": {"hotkey": ""}}}
    assert _extract_miner_meta_from_payload(payload) is None


def test_extract_miner_meta_from_payload_missing():
    assert _extract_miner_meta_from_payload({}) is None
    assert _extract_miner_meta_from_payload({"telemetry": {}}) is None


def test_build_winner_meta_from_uid_found():
    uid_to_hk = {5: "hk123", 10: "hk456"}
    miner_meta_by_hk = {
        "hk123": {"hotkey": "hk123", "chute_id": "c1", "slug": "s1"},
    }
    result = _build_winner_meta_from_uid(5, uid_to_hk, miner_meta_by_hk)
    assert result == {"hotkey": "hk123", "chute_id": "c1", "slug": "s1"}


def test_build_winner_meta_from_uid_no_meta():
    uid_to_hk = {5: "hk123"}
    miner_meta_by_hk = {}
    result = _build_winner_meta_from_uid(5, uid_to_hk, miner_meta_by_hk)
    assert result == {"hotkey": "hk123", "chute_id": None, "slug": None}


def test_build_winner_meta_from_uid_none():
    assert _build_winner_meta_from_uid(None, {}, {}) is None


def test_weighted_median_simple():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    assert _weighted_median(values, weights) == 2.0


def test_weighted_median_weighted():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 0.0, 1.0]
    result = _weighted_median(values, weights)
    assert result == 1.0


def test_weighted_median_single_value():
    values = [5.0]
    weights = [1.0]
    assert _weighted_median(values, weights) == 5.0


def test_days_to_blocks_valid():
    assert _days_to_blocks(1) == 7200
    assert _days_to_blocks(2) == 14400
    assert _days_to_blocks(0.5) == 3600


def test_days_to_blocks_invalid():
    assert _days_to_blocks(None) is None
    assert _days_to_blocks(0) is None
    assert _days_to_blocks(-1) is None


def test_stake_of_found():
    stake_by_hk = {"hk1": 100.5, "hk2": 200.0}
    assert _stake_of("hk1", stake_by_hk) == 100.5
    assert _stake_of("hk2", stake_by_hk) == 200.0


def test_stake_of_not_found():
    stake_by_hk = {"hk1": 100.5}
    assert _stake_of("hk_unknown", stake_by_hk) == 0.0


def test_stake_of_negative():
    stake_by_hk = {"hk1": -50.0}
    assert _stake_of("hk1", stake_by_hk) == 0.0


def test_extract_challenge_id_from_payload_task_id():
    assert _extract_challenge_id_from_payload({"task_id": "123"}) == "123"
    assert _extract_challenge_id_from_payload({"meta": {"task_id": "456"}}) == "456"
    assert _extract_challenge_id_from_payload({"telemetry": {"task_id": "789"}}) == "789"


def test_extract_challenge_id_from_payload_challenge_id():
    assert _extract_challenge_id_from_payload({"challenge_id": "chal_1"}) == "chal_1"
    assert _extract_challenge_id_from_payload({"meta": {"challenge_id": "chal_2"}}) == "chal_2"


def test_extract_challenge_id_from_payload_not_found():
    assert _extract_challenge_id_from_payload({}) is None
    assert _extract_challenge_id_from_payload({"other": "data"}) is None


def test_are_similar_by_challenges_similar():
    scores1 = {"c1": 0.8, "c2": 0.85, "c3": 0.9, "c4": 0.82, "c5": 0.88}
    scores2 = {"c1": 0.81, "c2": 0.84, "c3": 0.91, "c4": 0.83, "c5": 0.87}
    assert _are_similar_by_challenges(
        scores1, scores2, delta_abs=0.05, delta_rel=0.1, min_common_challenges=5
    )


def test_are_similar_by_challenges_not_similar():
    scores1 = {"c1": 0.8, "c2": 0.85, "c3": 0.9, "c4": 0.82, "c5": 0.88}
    scores2 = {"c1": 0.5, "c2": 0.84, "c3": 0.91, "c4": 0.83, "c5": 0.87}
    assert not _are_similar_by_challenges(
        scores1, scores2, delta_abs=0.05, delta_rel=0.1, min_common_challenges=5
    )


def test_are_similar_by_challenges_insufficient_common():
    scores1 = {"c1": 0.8, "c2": 0.85}
    scores2 = {"c1": 0.81, "c2": 0.84}
    assert not _are_similar_by_challenges(
        scores1, scores2, delta_abs=0.05, delta_rel=0.1, min_common_challenges=5
    )


def test_extract_elements_from_manifest_with_objects():
    manifest = SimpleNamespace(
        elements=[
            SimpleNamespace(element_id="e1", weight=0.5, eval_window=2),
            SimpleNamespace(id="e2", weight=0.3),
        ]
    )
    result = _extract_elements_from_manifest(manifest)
    assert len(result) == 2
    assert result[0] == ("e1", 0.5, 2)
    assert result[1] == ("e2", 0.3, None)


def test_extract_elements_from_manifest_with_dicts():
    manifest = SimpleNamespace(
        elements=[
            {"element_id": "e1", "weight": 0.5, "eval_window": 3},
            {"id": "e2", "weight": 0.3},
        ]
    )
    result = _extract_elements_from_manifest(manifest)
    assert len(result) == 2
    assert result[0] == ("e1", 0.5, 3)
    assert result[1] == ("e2", 0.3, None)


def test_extract_elements_from_manifest_empty():
    manifest = SimpleNamespace(elements=[])
    assert _extract_elements_from_manifest(manifest) == []

    manifest = SimpleNamespace(elements=None)
    assert _extract_elements_from_manifest(manifest) == []

