import pytest
from pydantic import ValidationError
from scorevision.utils.schemas import (
    ChallengeFrame,
    CricketDeliveryPrediction,
    ChallengeRequest,
    ChallengeResponse,
    FramePrediction,
    PredictionPayload,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)


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


def test_challenge_request_valid_with_target_frames():
    req = ChallengeRequest(
        challenge_id="abc",
        video_url="https://example.com/snooker.mp4",
        target_frames=[50, 150, 250, 350, 450],
    )
    assert req.target_frames == [50, 150, 250, 350, 450]


def test_challenge_request_rejects_negative_target_frame():
    with pytest.raises(ValidationError):
        ChallengeRequest(
            challenge_id="abc",
            video_url="https://example.com/snooker.mp4",
            target_frames=[50, -1],
        )


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


def test_challenge_response_valid():
    resp = ChallengeResponse(
        challenge_id="abc",
        predictions=[FramePrediction(frame=10, action="pass")],
        processing_time=1.5,
    )
    assert len(resp.predictions or []) == 1


def test_challenge_response_cricket_valid():
    resp = ChallengeResponse(
        challenge_id="abc",
        prediction=CricketDeliveryPrediction(kph=132.0, bounce_x=6.5),
        processing_time=0.0,
    )
    assert resp.prediction is not None
    assert resp.prediction_count == 1


def test_challenge_response_snooker_ball_state_valid():
    resp = ChallengeResponse(
        challenge_id="abc",
        prediction=SnookerBallStatePrediction(
            frames=[
                SnookerBallStateFrame(
                    frame=10,
                    balls=[
                        SnookerBallPrediction(
                            label="cue",
                            x=0.5,
                            y=0.5,
                            state="on_table",
                        )
                    ],
                )
            ]
        ),
        processing_time=0.2,
    )
    assert isinstance(resp.prediction, PredictionPayload)
    assert resp.prediction.type == "snooker_ball_state"
    assert resp.prediction_count == 1


def test_challenge_response_requires_exactly_one_payload():
    with pytest.raises(ValidationError):
        ChallengeResponse(
            challenge_id="abc",
            processing_time=0.0,
        )
