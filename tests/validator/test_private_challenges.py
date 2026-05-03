import pytest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from scorevision.validator.central.private_track.challenges import (
    get_challenge_with_ground_truth,
    parse_ground_truth_payload,
)
from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction


_FAKE_SETTINGS = SimpleNamespace(
    PRIVATE_MIN_ACTIONS_FOR_CHALLENGE=3,
    PRIVATE_GT_API_URL="https://gt.example.com",
)

_MODULE = "scorevision.validator.central.private_track.challenges"


def _patch_settings():
    return patch(f"{_MODULE}.get_settings", return_value=_FAKE_SETTINGS)


def test_parse_ground_truth_payload_returns_cricket_row_from_dict():
    parsed = parse_ground_truth_payload(
        {
            "r2_url": "https://example.com/ignored.csv",
            "inningsid": 1,
            "overid": 2,
            "ball_in_over": 3,
            "release_y": -0.45,
        }
    )
    assert isinstance(parsed, CricketDeliveryPrediction)
    assert parsed.overid == 2
    assert "r2_url" not in parsed.model_dump(mode="json")


def test_parse_ground_truth_payload_accepts_cricket_aliases_from_dict():
    parsed = parse_ground_truth_payload(
        {
            "innings": 1,
            "over": 2,
            "ball": 3,
            "overs": "1.3",
            "rel_y": -0.45,
            "rel_z": 1.9,
            "inter_d": 6.1,
            "swing_deg": -1.2,
            "deviation_deg": 0.4,
            "wkts": 0,
        }
    )
    assert isinstance(parsed, CricketDeliveryPrediction)
    assert parsed.inningsid == 1
    assert parsed.overid == 2
    assert parsed.ball_in_over == 3
    assert parsed.scorecard_overs == "1.3"


def test_parse_ground_truth_payload_returns_cricket_row_from_singleton_list():
    parsed = parse_ground_truth_payload(
        [{
            "inningsid": 1,
            "overid": 2,
            "ball_in_over": 3,
            "release_y": -0.45,
        }]
    )
    assert isinstance(parsed, CricketDeliveryPrediction)
    assert parsed.ball_in_over == 3


@pytest.mark.asyncio
async def test_get_challenge_returns_none_when_actions_insufficient():
    fake_chal = {"task_id": "c123", "video_url": "https://example.com/v1.mp4"}
    fake_gt = [FramePrediction(frame=25, action="pass")]

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
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
    assert len(challenge.payload_frames) == 2


@pytest.mark.asyncio
async def test_get_challenge_accepts_cricket_ground_truth_without_min_action_gate():
    fake_chal = {"task_id": "c999", "video_url": "https://example.com/cricket.mp4"}
    fake_gt = CricketDeliveryPrediction(
        inningsid=1,
        overid=1,
        ball_in_over=1,
        release_y=-0.49,
    )

    with _patch_settings(), \
         patch(f"{_MODULE}.fetch_next_challenge", new_callable=AsyncMock, return_value=fake_chal), \
         patch(f"{_MODULE}.fetch_ground_truth", new_callable=AsyncMock, return_value=fake_gt):
        challenge = await get_challenge_with_ground_truth(
            manifest_hash="abc123",
            element_id="manak0/Element-CricketBallTrack",
            keypair=None,
            max_retries=1,
        )

    assert challenge is not None
    assert isinstance(challenge.ground_truth, CricketDeliveryPrediction)
