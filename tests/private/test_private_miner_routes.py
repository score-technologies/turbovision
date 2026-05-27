from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from scorevision.miner.private_track.routes import handle_challenge
from scorevision.utils.schemas import (
    ChallengeRequest,
    CricketDeliveryPrediction,
    FramePrediction,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)


@pytest.mark.asyncio
async def test_handle_challenge_soccer_mode_returns_predictions():
    request = ChallengeRequest(challenge_id="c1", video_url="https://example.com/v.mp4")
    fake_video_path = Path("/tmp/fake.mp4")

    with (
        patch("scorevision.miner.private_track.routes.MINER_MODE", "soccer_action"),
        patch("scorevision.miner.private_track.routes.download_video", new=AsyncMock(return_value=fake_video_path)),
        patch(
            "scorevision.miner.private_track.routes.predict_actions",
            return_value=[FramePrediction(frame=25, action="pass")],
        ),
        patch("scorevision.miner.private_track.routes.delete_video"),
    ):
        response = await handle_challenge(request)

    assert response.predictions is not None
    assert response.prediction is not None
    assert response.prediction.type == "soccer_action"
    assert len(response.predictions) == 1


@pytest.mark.asyncio
async def test_handle_challenge_cricket_mode_returns_prediction():
    request = ChallengeRequest(challenge_id="c1", video_url="https://example.com/v.mp4")

    with (
        patch("scorevision.miner.private_track.routes.MINER_MODE", "cricket_delivery"),
        patch(
            "scorevision.miner.private_track.routes.predict_cricket_delivery",
            return_value=CricketDeliveryPrediction(kph=130.0, bounce_x=6.0, stump_y=0.2),
        ),
    ):
        response = await handle_challenge(request)

    assert response.predictions is None
    assert response.prediction is not None
    assert response.prediction.type == "cricket_delivery"
    assert response.prediction.item is not None
    assert response.prediction.item.kph == 130.0


@pytest.mark.asyncio
async def test_handle_challenge_snooker_mode_returns_prediction():
    request = ChallengeRequest(
        challenge_id="c1",
        video_url="https://example.com/v.mp4",
        target_frames=[50, 150],
    )

    with (
        patch("scorevision.miner.private_track.routes.MINER_MODE", "snooker_ball_state"),
        patch(
            "scorevision.miner.private_track.routes.predict_snooker_ball_state",
            return_value=SnookerBallStatePrediction(
                frames=[
                    SnookerBallStateFrame(
                        frame=0,
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
        ),
    ):
        response = await handle_challenge(request)

    assert response.predictions is None
    assert response.prediction is not None
    assert response.prediction.type == "snooker_ball_state"
    assert response.prediction.frames is not None
    assert response.prediction.frames[0].balls[0].label == "cue"


@pytest.mark.asyncio
async def test_handle_challenge_snooker_stub_uses_target_frame():
    request = ChallengeRequest(
        challenge_id="c1",
        video_url="https://example.com/v.mp4",
        target_frames=[50, 150],
    )

    with patch("scorevision.miner.private_track.routes.MINER_MODE", "snooker_ball_state"):
        response = await handle_challenge(request)

    assert response.prediction is not None
    assert response.prediction.frames is not None
    assert response.prediction.frames[0].frame == 50
