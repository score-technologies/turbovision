import importlib
from pathlib import Path
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest

from scorevision.miner.private_track.predictor import is_cricket_request, predict_cricket_delivery
from scorevision.utils.schemas import (
    ChallengeFrame,
    ChallengeRequest,
    CricketDeliveryPrediction,
    FramePrediction,
)


fiber = types.ModuleType("fiber")
fiber.constants = types.SimpleNamespace(
    MINER_HOTKEY="miner-hotkey",
    NONCE="nonce",
    SIGNATURE="signature",
    VALIDATOR_HOTKEY="validator-hotkey",
)
fiber.utils = types.SimpleNamespace(construct_header_signing_message=Mock(return_value="message"))
fiber_chain = types.ModuleType("fiber.chain")
fiber_chain.signatures = types.SimpleNamespace(
    get_hash=Mock(return_value="hash"),
    verify_signature=Mock(return_value=True),
)
fiber_miner = types.ModuleType("fiber.miner")
fiber_miner_security = types.ModuleType("fiber.miner.security")
fiber_nonce_management = types.ModuleType("fiber.miner.security.nonce_management")
fiber_nonce_management.NonceManager = Mock(return_value=Mock(nonce_is_valid=Mock(return_value=True)))
sys.modules.setdefault("fiber", fiber)
sys.modules.setdefault("fiber.chain", fiber_chain)
sys.modules.setdefault("fiber.miner", fiber_miner)
sys.modules.setdefault("fiber.miner.security", fiber_miner_security)
sys.modules.setdefault("fiber.miner.security.nonce_management", fiber_nonce_management)

routes = importlib.import_module("scorevision.miner.private_track.routes")


def test_is_cricket_request_uses_current_schema_context():
    assert is_cricket_request(
        ChallengeRequest(
            challenge_id="task-1",
            video_url="https://example.com/assets/cricket-delivery.mp4",
        )
    )
    assert is_cricket_request(
        ChallengeRequest(
            challenge_id="task-2",
            frames=[ChallengeFrame(frame_id=0, url="https://example.com/cricket/frame.jpg")],
        )
    )
    assert not is_cricket_request(
        ChallengeRequest(
            challenge_id="task-3",
            video_url="https://example.com/assets/football-action.mp4",
        )
    )


def test_predict_cricket_delivery_returns_canonical_dummy_shape():
    prediction = predict_cricket_delivery(
        ChallengeRequest(
            challenge_id="cricket-task",
            video_url="https://example.com/cricket.mp4",
        )
    )

    assert isinstance(prediction, CricketDeliveryPrediction)
    assert prediction.model_dump(mode="json") == {
        "match": "dummy-cricket-stub",
        "matchid": -1,
        "inningsid": -1,
        "overid": -1,
        "ball_in_over": -1,
        "ballid": -1,
        "xlsx_overs": "stub",
        "scorecard_overs": "stub",
        "kph": -1.0,
        "release_y": -999.0,
        "release_z": -999.0,
        "bounce_x": -999.0,
        "bounce_y": -999.0,
        "impact_x": -999.0,
        "impact_y": -999.0,
        "impact_z": -999.0,
        "interception_distance": -999.0,
        "stump_y": -999.0,
        "stump_z": -999.0,
        "swing_angle": -999.0,
        "deviation": -999.0,
        "runs": -1,
        "wickets": -1,
    }


@pytest.mark.asyncio
async def test_handle_challenge_returns_cricket_response_without_downloading(monkeypatch):
    download_video = AsyncMock(side_effect=AssertionError("cricket stub should not download video"))
    monkeypatch.setattr(routes, "download_video", download_video)

    response = await routes.handle_challenge(
        ChallengeRequest(
            challenge_id="cricket-task",
            video_url="https://example.com/cricket.mp4",
        )
    )

    assert response.challenge_id == "cricket-task"
    assert response.predictions is None
    assert isinstance(response.prediction, CricketDeliveryPrediction)
    assert response.prediction.kph == -1.0
    assert response.model_dump(mode="json")["prediction"]["bounce_x"] == -999.0
    download_video.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_challenge_preserves_legacy_action_response(monkeypatch):
    video_path = Path("/tmp/legacy-action.mp4")
    predictions = [FramePrediction(frame=25, action="pass", confidence=0.8)]
    download_video = AsyncMock(return_value=video_path)
    predict_actions = Mock(return_value=predictions)
    delete_video = Mock()
    monkeypatch.setattr(routes, "download_video", download_video)
    monkeypatch.setattr(routes, "predict_actions", predict_actions)
    monkeypatch.setattr(routes, "delete_video", delete_video)

    response = await routes.handle_challenge(
        ChallengeRequest(
            challenge_id="football-task",
            video_url="https://example.com/football.mp4",
        )
    )

    assert response.challenge_id == "football-task"
    assert response.predictions == predictions
    assert response.prediction is None
    assert response.model_dump(mode="json")["predictions"] == [
        {"frame": 25, "action": "pass", "confidence": 0.8}
    ]
    download_video.assert_awaited_once_with("https://example.com/football.mp4")
    predict_actions.assert_called_once_with(video_path)
    delete_video.assert_called_once_with(video_path)
