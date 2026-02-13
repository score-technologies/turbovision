import pytest
from pydantic import ValidationError
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse, FramePrediction


def test_frame_prediction_valid():
    pred = FramePrediction(frame=42, action="pass")
    assert pred.frame == 42
    assert pred.action == "pass"


def test_frame_prediction_negative_frame_rejected():
    with pytest.raises(ValidationError):
        FramePrediction(frame=-1, action="pass")


def test_challenge_request_valid():
    req = ChallengeRequest(challenge_id="abc", video_url="https://example.com/v.mp4")
    assert req.challenge_id == "abc"


def test_challenge_response_valid():
    resp = ChallengeResponse(
        challenge_id="abc",
        predictions=[FramePrediction(frame=10, action="pass")],
        processing_time=1.5,
    )
    assert len(resp.predictions) == 1


def test_challenge_response_empty_predictions():
    resp = ChallengeResponse(
        challenge_id="abc",
        processing_time=0.0,
    )
    assert resp.predictions == []
