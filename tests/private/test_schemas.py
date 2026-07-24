import pytest
from pydantic import ValidationError
from scorevision.utils.schemas import (
    ChallengeFrame,
    CricketDeliveryPrediction,
    ChallengeRequest,
    ChallengeResponse,
    FramePrediction,
    TCGGradingPrediction,
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


def test_challenge_request_valid_with_frames_only():
    req = ChallengeRequest(
        challenge_id="abc",
        frames=[ChallengeFrame(frame_id=0, url="https://example.com/f0.jpg")],
    )
    assert req.challenge_id == "abc"
    assert req.video_url is None
    assert req.frames is not None


def test_challenge_request_valid_with_image_only():
    req = ChallengeRequest(
        challenge_id="abc",
        image_url="https://example.com/card.png",
    )
    assert req.image_url == "https://example.com/card.png"
    assert req.video_url is None


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


def test_challenge_response_tcg_grading_uses_public_schema():
    resp = ChallengeResponse(
        challenge_id="abc",
        prediction={
            "Header": {"card_grade": 7},
            "Grading_Features": {
                "subgrade_surface": 6,
                "subgrade_centering": 10,
                "subgrade_edges": 10,
                "subgrade_corners": 10,
            },
        },
        processing_time=2.5,
    )

    assert isinstance(resp.prediction, TCGGradingPrediction)
    assert resp.prediction_count == 5
    assert resp.model_dump(mode="json") == {
        "challenge_id": "abc",
        "prediction": {
            "Header": {"card_grade": 7},
            "Grading_Features": {
                "subgrade_surface": 6,
                "subgrade_centering": 10,
                "subgrade_edges": 10,
                "subgrade_corners": 10,
            },
        },
        "processing_time": 2.5,
    }


def test_tcg_grading_response_rounds_all_grades_down():
    resp = ChallengeResponse(
        challenge_id="abc",
        prediction={
            "Header": {"card_grade": 8.9},
            "Grading_Features": {
                "subgrade_surface": 8.5,
                "subgrade_centering": 9.99,
                "subgrade_edges": 7.1,
                "subgrade_corners": 10,
            },
        },
        processing_time=1.0,
    )

    assert isinstance(resp.prediction, TCGGradingPrediction)
    assert resp.prediction.Header.card_grade == 8
    assert resp.prediction.Grading_Features.subgrade_surface == 8
    assert resp.prediction.Grading_Features.subgrade_centering == 9
    assert resp.prediction.Grading_Features.subgrade_edges == 7
    assert resp.prediction.Grading_Features.subgrade_corners == 10


def test_challenge_response_requires_exactly_one_payload():
    with pytest.raises(ValidationError):
        ChallengeResponse(
            challenge_id="abc",
            processing_time=0.0,
        )
