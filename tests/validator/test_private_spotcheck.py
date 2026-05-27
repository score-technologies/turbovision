from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scorevision.utils.schemas import (
    CricketDeliveryPrediction,
    FramePrediction,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)
from scorevision.validator.audit.private_track import spotcheck as spotcheck_mod


def test_infer_groundtruth_type_from_explicit_field():
    challenge_results = [{"groundtruth_type": "cricket_delivery"}]
    miner_responses = {}
    assert (
        spotcheck_mod._infer_groundtruth_type(challenge_results, miner_responses)
        == "cricket_delivery"
    )


def test_infer_groundtruth_type_from_explicit_snooker_field():
    challenge_results = [{"groundtruth_type": "snooker_ball_state"}]
    miner_responses = {}
    assert (
        spotcheck_mod._infer_groundtruth_type(challenge_results, miner_responses)
        == "snooker_ball_state"
    )


def test_infer_groundtruth_type_from_prediction_shape():
    challenge_results = [{"miner_hotkey": "hk1"}]
    miner_responses = {"hk1": [{"kph": 129.2, "bounce_x": 5.8}]}
    assert (
        spotcheck_mod._infer_groundtruth_type(challenge_results, miner_responses)
        == "cricket_delivery"
    )


def test_infer_groundtruth_type_from_snooker_prediction_shape():
    challenge_results = [{"miner_hotkey": "hk1"}]
    miner_responses = {"hk1": [{"frame": 10, "balls": []}]}
    assert (
        spotcheck_mod._infer_groundtruth_type(challenge_results, miner_responses)
        == "snooker_ball_state"
    )


def test_rescore_miner_soccer():
    predictions = [{"frame": 25, "action": "pass"}]
    gt = [FramePrediction(frame=25, action="pass")]
    score = spotcheck_mod.rescore_miner_soccer(predictions, gt)
    assert 0.0 <= score <= 1.0


def test_rescore_miner_cricket():
    predictions = [{"kph": 128.0, "bounce_x": 6.0, "stump_y": 0.1}]
    gt = CricketDeliveryPrediction(kph=128.0, bounce_x=6.0, stump_y=0.1)
    score = spotcheck_mod.rescore_miner_cricket(predictions, gt)
    assert 0.0 <= score <= 1.0


def test_rescore_miner_snooker():
    predictions = [
        {
            "frame": 0,
            "balls": [
                {"label": "cue", "x": 0.5, "y": 0.5, "state": "on_table"},
                {"label": "red", "x": 0.6, "y": 0.6, "state": "on_table"},
            ],
        }
    ]
    gt = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=0,
                balls=[
                    SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.6, y=0.6, state="on_table"),
                ],
            )
        ]
    )
    score = spotcheck_mod.rescore_miner_snooker(predictions, gt)
    assert score == 1.0


@pytest.mark.asyncio
async def test_run_private_spotcheck_cricket_passes_element_and_groundtruth_type():
    keypair = object()
    threshold = 0.95
    challenge_id = "chal-1"
    challenge_results = [
        {
            "element_id": "manako/DetectCricketDelivery",
            "miner_hotkey": "hk1",
            "score": 0.8,
        }
    ]
    miner_responses = {"hk1": [{"kph": 128.0, "bounce_x": 6.0, "stump_y": 0.1}]}

    with (
        patch.object(
            spotcheck_mod,
            "fetch_miner_responses",
            new=AsyncMock(return_value=miner_responses),
        ),
        patch.object(
            spotcheck_mod,
            "fetch_ground_truth",
            new=AsyncMock(return_value=CricketDeliveryPrediction(kph=128.0, bounce_x=6.0, stump_y=0.1)),
        ) as fetch_gt_mock,
        patch.object(
            spotcheck_mod,
            "calculate_match_percentage",
            return_value=1.0,
        ),
    ):
        results = await spotcheck_mod.run_private_spotcheck(
            challenge_id=challenge_id,
            challenge_results=challenge_results,
            keypair=keypair,
            threshold=threshold,
        )

    assert len(results) == 1
    assert results[0].element_id == "manako/DetectCricketDelivery"
    fetch_gt_mock.assert_awaited_once_with(
        challenge_id,
        keypair,
        element_id="manako/DetectCricketDelivery",
        groundtruth_type="cricket_delivery",
    )


@pytest.mark.asyncio
async def test_run_private_spotcheck_snooker_passes_element_and_groundtruth_type():
    keypair = object()
    threshold = 0.95
    challenge_id = "chal-snooker-1"
    challenge_results = [
        {
            "element_id": "manako/DetectSnookerBallState",
            "groundtruth_type": "snooker_ball_state",
            "miner_hotkey": "hk1",
            "score": 1.0,
        }
    ]
    gt = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=0,
                balls=[
                    SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table"),
                ],
            )
        ]
    )
    miner_responses = {
        "hk1": [
            {
                "frame": 0,
                "balls": [
                    {"label": "cue", "x": 0.5, "y": 0.5, "state": "on_table"},
                ],
            }
        ]
    }

    with (
        patch.object(
            spotcheck_mod,
            "fetch_miner_responses",
            new=AsyncMock(return_value=miner_responses),
        ),
        patch.object(
            spotcheck_mod,
            "fetch_ground_truth",
            new=AsyncMock(return_value=gt),
        ) as fetch_gt_mock,
        patch.object(
            spotcheck_mod,
            "calculate_match_percentage",
            return_value=1.0,
        ),
    ):
        results = await spotcheck_mod.run_private_spotcheck(
            challenge_id=challenge_id,
            challenge_results=challenge_results,
            keypair=keypair,
            threshold=threshold,
        )

    assert len(results) == 1
    assert results[0].element_id == "manako/DetectSnookerBallState"
    fetch_gt_mock.assert_awaited_once_with(
        challenge_id,
        keypair,
        element_id="manako/DetectSnookerBallState",
        groundtruth_type="snooker_ball_state",
    )

