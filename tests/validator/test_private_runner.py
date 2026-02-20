from unittest.mock import AsyncMock, patch
import pytest
from scorevision.utils.schemas import ChallengeResponse, FramePrediction
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.miners import ChallengeAttempt
from scorevision.validator.central.private_track.registry import RegisteredMiner
from scorevision.validator.central.private_track.runner import (
    _challenge_miner,
    _is_weight_eligible_result,
)


def _miner() -> RegisteredMiner:
    return RegisteredMiner(
        uid=7,
        hotkey="5MinerHotkey",
        ip="127.0.0.1",
        port=8000,
        image_repo="org/repo",
        image_tag="v1",
        commit_block=123,
    )


def _challenge() -> Challenge:
    return Challenge(
        challenge_id="challenge-1",
        video_url="https://example.com/video.mp4",
        ground_truth=[FramePrediction(frame=25, action="pass")],
    )


@pytest.mark.asyncio
async def test_challenge_miner_scores_when_response_is_on_time():
    attempt = ChallengeAttempt(
        response=ChallengeResponse(
            challenge_id="challenge-1",
            predictions=[FramePrediction(frame=25, action="pass")],
            processing_time=1.2,
        ),
        elapsed_s=2.5,
        timed_out=False,
    )

    with (
        patch(
            "scorevision.validator.central.private_track.runner.send_challenge",
            new=AsyncMock(return_value=attempt),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.score_predictions",
            return_value=0.83,
        ) as score_mock,
    ):
        result = await _challenge_miner(
            miner=_miner(),
            challenge=_challenge(),
            keypair=None,
            timeout=30.0,
            block=1234,
        )

    score_mock.assert_called_once()
    assert result["score"] == 0.83
    assert result["prediction_count"] == 1
    assert result["timed_out"] is False
    assert result["processing_time"] == 2.5
    assert result["response_time_s"] == 2.5


@pytest.mark.asyncio
async def test_challenge_miner_excludes_timeout_from_weights():
    attempt = ChallengeAttempt(
        response=None,
        elapsed_s=30.2,
        timed_out=True,
    )

    with (
        patch(
            "scorevision.validator.central.private_track.runner.send_challenge",
            new=AsyncMock(return_value=attempt),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.score_predictions",
            return_value=0.91,
        ) as score_mock,
    ):
        result = await _challenge_miner(
            miner=_miner(),
            challenge=_challenge(),
            keypair=None,
            timeout=30.0,
            block=1234,
        )

    score_mock.assert_not_called()
    assert result["score"] == 0.0
    assert result["prediction_count"] == 0
    assert result["timed_out"] is True


def test_is_weight_eligible_result_defaults_to_true():
    assert _is_weight_eligible_result({"miner_hotkey": "hk"}) is True


def test_is_weight_eligible_result_false_for_timeout():
    assert _is_weight_eligible_result({"timed_out": True}) is False
