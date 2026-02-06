import pytest
from scorevision.validator.audit import (
    calculate_match_percentage,
    scores_match,
    calculate_next_spotcheck_delay,
)
from scorevision.validator.models import SpotcheckResult, ChallengeRecord


def test_calculate_match_percentage_identical():
    assert calculate_match_percentage(0.8, 0.8) == 1.0
    assert calculate_match_percentage(1.0, 1.0) == 1.0
    assert calculate_match_percentage(0.0, 0.0) == 1.0


def test_calculate_match_percentage_similar():
    result = calculate_match_percentage(0.8, 0.76)
    assert 0.94 < result < 0.96


def test_calculate_match_percentage_different():
    result = calculate_match_percentage(0.8, 0.4)
    assert result == 0.5


def test_calculate_match_percentage_one_zero():
    assert calculate_match_percentage(0.8, 0.0) == 0.0
    assert calculate_match_percentage(0.0, 0.8) == 0.0


def test_scores_match_threshold():
    assert scores_match(0.8, 0.8, 0.95) is True
    assert scores_match(0.8, 0.76, 0.95) is True
    assert scores_match(0.8, 0.5, 0.95) is False


def test_scores_match_exact_threshold():
    assert scores_match(0.8, 0.76, 0.95) is True
    assert scores_match(0.8, 0.68, 0.95) is False


def test_calculate_next_spotcheck_delay_range():
    for _ in range(100):
        delay = calculate_next_spotcheck_delay(7200, 14400)
        assert 7200 <= delay <= 14400


def test_calculate_next_spotcheck_delay_same_bounds():
    delay = calculate_next_spotcheck_delay(1000, 1000)
    assert delay == 1000


def test_spotcheck_result_dataclass():
    result = SpotcheckResult(
        challenge_id="test-123",
        element_id="soccer_detect",
        miner_hotkey="hk123",
        central_score=0.85,
        audit_score=0.83,
        match_percentage=0.976,
        passed=True,
    )
    assert result.challenge_id == "test-123"
    assert result.passed is True
    assert result.match_percentage == 0.976


def test_challenge_record_dataclass():
    record = ChallengeRecord(
        challenge_id="chal-123",
        element_id="soccer_detect",
        window_id="2025-W05",
        block=1000000,
        miner_hotkey="hk_abc",
        central_score=0.88,
        payload={"task_id": 123},
    )
    assert record.challenge_id == "chal-123"
    assert record.element_id == "soccer_detect"
    assert record.central_score == 0.88
    assert record.payload["task_id"] == 123
    assert record.responses_key is None


def test_challenge_record_with_responses_key():
    record = ChallengeRecord(
        challenge_id="chal-456",
        element_id="soccer_detect",
        window_id="2025-W05",
        block=1000000,
        miner_hotkey="hk_abc",
        central_score=0.88,
        payload={},
        responses_key="scorevision/elem/hk123/responses/00123-abc.json",
    )
    assert record.responses_key == "scorevision/elem/hk123/responses/00123-abc.json"


def test_challenge_record_with_scored_frame_numbers():
    record = ChallengeRecord(
        challenge_id="chal-789",
        element_id="soccer_detect",
        window_id="2025-W05",
        block=1000000,
        miner_hotkey="hk_abc",
        central_score=0.92,
        payload={},
        scored_frame_numbers=[10, 11, 12, 13, 14],
    )
    assert record.scored_frame_numbers == [10, 11, 12, 13, 14]


def test_challenge_record_scored_frame_numbers_default_none():
    record = ChallengeRecord(
        challenge_id="chal-default",
        element_id="soccer_detect",
        window_id="2025-W05",
        block=1000000,
        miner_hotkey="hk_abc",
        central_score=0.85,
        payload={},
    )
    assert record.scored_frame_numbers is None
