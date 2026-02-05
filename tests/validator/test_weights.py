import pytest
from scorevision.validator.payload import (
    extract_miner_and_score,
    extract_miner_meta,
    build_winner_meta,
    extract_challenge_id,
    extract_elements_from_manifest,
)
from scorevision.validator.scoring import (
    weighted_median,
    days_to_blocks,
    stake_of,
    are_similar_by_challenges,
)
from scorevision.validator.models import WeightsResult, MinerMeta


def test_extract_miner_and_score_from_payload_valid():
    payload = {
        "telemetry": {"miner": {"hotkey": "hk123"}},
        "metrics": {"composite_score": 0.85},
    }
    hk_to_uid = {"hk123": 5}
    uid, score = extract_miner_and_score(payload, hk_to_uid)
    assert uid == 5
    assert score == 0.85


def test_extract_miner_and_score_from_payload_unknown_hotkey():
    payload = {
        "telemetry": {"miner": {"hotkey": "unknown"}},
        "metrics": {"composite_score": 0.85},
    }
    hk_to_uid = {"hk123": 5}
    uid, score = extract_miner_and_score(payload, hk_to_uid)
    assert uid is None
    assert score is None


def test_extract_miner_meta_from_payload_valid():
    payload = {
        "telemetry": {
            "miner": {"hotkey": "hk123", "chute_id": "chute1", "slug": "slug1"}
        },
    }
    meta = extract_miner_meta(payload)
    assert meta.hotkey == "hk123"
    assert meta.chute_id == "chute1"
    assert meta.slug == "slug1"


def test_build_winner_meta_from_uid_found():
    uid_to_hk = {5: "hk123"}
    miner_meta_by_hk = {"hk123": MinerMeta(hotkey="hk123", chute_id="c1", slug="s1")}
    meta = build_winner_meta(5, uid_to_hk, miner_meta_by_hk)
    assert meta["hotkey"] == "hk123"
    assert meta["chute_id"] == "c1"


def test_build_winner_meta_from_uid_none():
    meta = build_winner_meta(None, {}, {})
    assert meta is None


def test_weighted_median_simple():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    result = weighted_median(values, weights)
    assert result == 2.0


def test_weighted_median_weighted():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 0.0, 1.0]
    result = weighted_median(values, weights)
    assert result == 1.0


def test_stake_of_found():
    stake_by_hk = {"hk123": 1000.0}
    assert stake_of("hk123", stake_by_hk) == 1000.0


def test_stake_of_not_found():
    stake_by_hk = {"hk123": 1000.0}
    assert stake_of("unknown", stake_by_hk) == 0.0


def test_stake_of_negative():
    stake_by_hk = {"hk123": -100.0}
    assert stake_of("hk123", stake_by_hk) == 0.0


def test_extract_challenge_id_from_payload_task_id():
    payload = {"meta": {"task_id": "task123"}}
    assert extract_challenge_id(payload) == "task123"


def test_extract_challenge_id_from_payload_challenge_id():
    payload = {"challenge_id": "chal456"}
    assert extract_challenge_id(payload) == "chal456"


def test_extract_challenge_id_from_payload_not_found():
    payload = {"other": "data"}
    assert extract_challenge_id(payload) is None


def test_are_similar_by_challenges_similar():
    scores1 = {"c1": 0.8, "c2": 0.7, "c3": 0.9, "c4": 0.85, "c5": 0.75}
    scores2 = {"c1": 0.8, "c2": 0.7, "c3": 0.9, "c4": 0.85, "c5": 0.75}
    assert are_similar_by_challenges(scores1, scores2, delta_abs=0.01, delta_rel=0.01) is True


def test_are_similar_by_challenges_not_similar():
    scores1 = {"c1": 0.8, "c2": 0.7, "c3": 0.9, "c4": 0.85, "c5": 0.75}
    scores2 = {"c1": 0.5, "c2": 0.4, "c3": 0.6, "c4": 0.55, "c5": 0.45}
    assert are_similar_by_challenges(scores1, scores2, delta_abs=0.01, delta_rel=0.01) is False


def test_are_similar_by_challenges_insufficient_common():
    scores1 = {"c1": 0.8, "c2": 0.7}
    scores2 = {"c3": 0.8, "c4": 0.7}
    assert are_similar_by_challenges(scores1, scores2, delta_abs=0.01, delta_rel=0.01, min_common_challenges=5) is False


def test_weights_result_dataclass():
    result = WeightsResult(
        element_id="soccer_detect",
        window_id="2025-W05",
        winner_uid=7,
        scores_by_uid={7: 0.9, 12: 0.85},
        winner_meta={"hotkey": "hk7", "chute_id": "c1", "slug": "s1"},
    )
    assert result.element_id == "soccer_detect"
    assert result.winner_uid == 7
    assert result.scores_by_uid[7] == 0.9
