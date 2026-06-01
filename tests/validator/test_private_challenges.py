import pytest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from scorevision.validator.central.private_track.challenges import (
    get_challenge_with_ground_truth,
)
from scorevision.utils.schemas import (
    CricketDeliveryPrediction,
    FramePrediction,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)


_FAKE_SETTINGS = SimpleNamespace(
    PRIVATE_MIN_ACTIONS_FOR_CHALLENGE=3,
    PRIVATE_GT_API_URL="https://gt.example.com",
)

_MODULE = "scorevision.validator.central.private_track.challenges"


def _patch_settings():
    return patch(f"{_MODULE}.get_settings", return_value=_FAKE_SETTINGS)


@pytest.mark.asyncio
async def test_get_challenge_returns_none_when_actions_insufficient():
    fake_chal = {"task_id": "c123", "video_url": "https://example.com/v1.mp4"}
    fake_gt = [FramePrediction(frame=25, action="pass")]

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", new_callable=AsyncMock), \
         patch(f"{_MODULE}.fetch_ground_truth", new_callable=AsyncMock, return_value=fake_gt):

        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            max_retries=1,
        )

        assert challenge is None


@pytest.mark.asyncio
async def test_get_challenge_retries_on_ground_truth_fetch_error():
    fake_chal = {"task_id": "c123", "video_url": "https://example.com/v1.mp4"}
    fake_gt = [
        FramePrediction(frame=i * 25, action="pass") for i in range(5)
    ]

    fetch_gt_mock = AsyncMock(side_effect=[Exception("API down"), fake_gt])

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", new_callable=AsyncMock), \
         patch(f"{_MODULE}.fetch_ground_truth", fetch_gt_mock):

        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            max_retries=2,
        )

        assert challenge is not None
        assert challenge.challenge_id == "c123"
        assert fetch_gt_mock.call_count == 2


@pytest.mark.asyncio
async def test_get_challenge_accepts_payload_frames_without_video_url():
    fake_chal = {
        "task_id": "c123",
        "payload": {
            "frames": [
                {"frame_id": 0, "url": "https://example.com/f0.jpg"},
                {"frame_id": 1, "url": "https://example.com/f1.jpg"},
            ]
        },
    }
    fake_gt = [FramePrediction(frame=i * 25, action="pass") for i in range(5)]

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", new_callable=AsyncMock), \
         patch(f"{_MODULE}.fetch_ground_truth", new_callable=AsyncMock, return_value=fake_gt):
        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            max_retries=1,
        )

    assert challenge is not None
    assert challenge.challenge_id == "c123"
    assert challenge.video_url is None
    assert challenge.payload_frames is not None
    assert len(challenge.payload_frames) == 1
    assert challenge.payload_frames[0].frame_id == 1


@pytest.mark.asyncio
async def test_get_challenge_accepts_single_cricket_ground_truth():
    fake_chal = {"task_id": "c123", "video_url": "https://example.com/v1.mp4"}
    cricket_gt = CricketDeliveryPrediction(kph=126.8, bounce_x=8.0, stump_y=0.01, deviation=1.1)

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", new_callable=AsyncMock), \
         patch(f"{_MODULE}.fetch_ground_truth", new_callable=AsyncMock, return_value=cricket_gt):
        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            groundtruth_type="cricket_delivery",
            max_retries=1,
        )

    assert challenge is not None
    assert challenge.groundtruth_type == "cricket_delivery"


@pytest.mark.asyncio
async def test_get_challenge_accepts_snooker_ball_state_ground_truth():
    fake_chal = {
        "task_id": "c123",
        "payload": {
            "clip_url": "https://example.com/v1.mp4",
            "target_frames": [50, 150, 250, 350, 450],
        },
    }
    snooker_gt = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
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
    )

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", new_callable=AsyncMock), \
         patch(f"{_MODULE}.fetch_ground_truth", new_callable=AsyncMock, return_value=snooker_gt):
        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            groundtruth_type="snooker_ball_state",
            max_retries=1,
        )

    assert challenge is not None
    assert challenge.groundtruth_type == "snooker_ball_state"
    assert challenge.video_url == "https://example.com/v1.mp4"
    assert challenge.target_frames == [50, 150, 250, 350, 450]


@pytest.mark.asyncio
async def test_get_challenge_rejects_snooker_without_target_frames():
    fake_chal = {"task_id": "c123", "video_url": "https://example.com/v1.mp4"}
    snooker_gt = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
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
    )
    complete_mock = AsyncMock()
    fetch_gt_mock = AsyncMock(return_value=snooker_gt)

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.complete_task_assignment", complete_mock), \
         patch(f"{_MODULE}.fetch_ground_truth", fetch_gt_mock):
        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="elem1",
            keypair=None,
            groundtruth_type="snooker_ball_state",
            max_retries=1,
        )

    assert challenge is None
    complete_mock.assert_not_awaited()
    fetch_gt_mock.assert_not_awaited()
