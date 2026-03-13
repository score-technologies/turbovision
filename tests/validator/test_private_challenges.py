import pytest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from scorevision.validator.central.private_track.challenges import (
    Challenge,
    get_challenge_with_ground_truth,
)
from scorevision.utils.schemas import FramePrediction


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
