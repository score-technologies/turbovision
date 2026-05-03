import pytest
from pydantic import ValidationError
from scorevision.utils.schemas import (
    ChallengeFrame,
    ChallengeRequest,
    ChallengeResponse,
    CricketDeliveryPrediction,
    FramePrediction,
)


def test_frame_prediction_valid():
    pred = FramePrediction(frame=42, action="pass")
    assert pred.frame == 42
    assert pred.action == "pass"


def test_frame_prediction_negative_frame_rejected():
    with pytest.raises(ValidationError):
        FramePrediction(frame=-1, action="pass")


def test_cricket_delivery_prediction_accepts_aliases():
    pred = CricketDeliveryPrediction(
        innings=1,
        over=2,
        ball=3,
        overs="2.3",
        rel_y=-0.12,
        rel_z=1.95,
        inter_d=4.1,
        swing_deg=-2.3,
        deviation_deg=1.2,
        wkts=1,
    )
    assert pred.inningsid == 1
    assert pred.overid == 2
    assert pred.ball_in_over == 3
    assert pred.scorecard_overs == "2.3"
    assert pred.release_y == -0.12
    assert pred.release_z == 1.95
    assert pred.interception_distance == 4.1
    assert pred.swing_angle == -2.3
    assert pred.deviation == 1.2
    assert pred.wickets == 1


def test_challenge_request_valid():
    req = ChallengeRequest(challenge_id="abc", video_url="https://example.com/v.mp4")
    assert req.challenge_id == "abc"


def test_challenge_request_valid_with_frames_only():
    req = ChallengeRequest(
        challenge_id="abc",
        frames=[ChallengeFrame(frame_id=0, url="https://example.com/f0.jpg")],
    )
    assert req.challenge_id == "abc"
    assert req.video_url is None
    assert req.frames is not None


def test_challenge_request_rejects_empty_payload():
    with pytest.raises(ValidationError):
        ChallengeRequest(challenge_id="abc")


def test_challenge_response_valid_legacy_predictions():
    resp = ChallengeResponse(
        challenge_id="abc",
        predictions=[FramePrediction(frame=10, action="pass")],
        processing_time=1.5,
    )
    assert resp.prediction_count == 1
    assert resp.is_cricket is False


def test_challenge_response_accepts_empty_legacy_predictions():
    resp = ChallengeResponse(
        challenge_id="abc",
        predictions=[],
        processing_time=0.0,
    )
    assert resp.prediction_count == 0
    assert resp.is_cricket is False


def test_challenge_response_valid_cricket_prediction():
    resp = ChallengeResponse(
        challenge_id="abc",
        prediction=CricketDeliveryPrediction(inningsid=1, overid=1, ball_in_over=1),
        processing_time=0.8,
    )
    assert resp.prediction_count == 1
    assert resp.is_cricket is True
    assert resp.model_dump(mode="json") == {
        "challenge_id": "abc",
        "prediction": {
            "match": None,
            "matchid": None,
            "inningsid": 1,
            "overid": 1,
            "ball_in_over": 1,
            "ballid": None,
            "xlsx_overs": None,
            "scorecard_overs": None,
            "kph": None,
            "release_y": None,
            "release_z": None,
            "bounce_x": None,
            "bounce_y": None,
            "impact_x": None,
            "impact_y": None,
            "impact_z": None,
            "interception_distance": None,
            "stump_y": None,
            "stump_z": None,
            "swing_angle": None,
            "deviation": None,
            "runs": None,
            "wickets": None,
        },
        "processing_time": 0.8,
    }


def test_challenge_response_rejects_missing_payload():
    with pytest.raises(ValidationError):
        ChallengeResponse(
            challenge_id="abc",
            processing_time=0.0,
        )


def test_challenge_response_rejects_mixed_payloads():
    with pytest.raises(ValidationError):
        ChallengeResponse(
            challenge_id="abc",
            predictions=[FramePrediction(frame=10, action="pass")],
            prediction=CricketDeliveryPrediction(inningsid=1),
            processing_time=0.0,
        )
