from unittest.mock import patch

import pytest

from scorevision.utils.schemas import (
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner


def _miner() -> RegisteredMiner:
    return RegisteredMiner(
        uid=7,
        hotkey="5MinerHotkey",
        ip="127.0.0.1",
        port=8000,
        image_repo="org/repo",
        image_tag="v1",
        commit_block=123,
        image_digest="sha256:abc123",
    )


@pytest.mark.asyncio
async def test_send_challenge_forwards_target_frames():
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "challenge_id": "snooker-1",
                "prediction": {
                    "type": "snooker_ball_state",
                    "frames": [
                        {
                            "frame": 50,
                            "balls": [
                                {
                                    "label": "cue",
                                    "x": 0.5,
                                    "y": 0.5,
                                    "state": "on_table",
                                }
                            ],
                        }
                    ],
                },
                "processing_time": 0.05,
            }

    class _FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

    challenge = Challenge(
        challenge_id="snooker-1",
        video_url="https://example.com/snooker.mp4",
        target_frames=[50, 150, 250],
        ground_truth=SnookerBallStatePrediction(
            frames=[
                SnookerBallStateFrame(
                    frame=50,
                    balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5)],
                )
            ]
        ),
        groundtruth_type="snooker_ball_state",
    )

    with (
        patch("scorevision.validator.central.private_track.miners.httpx.AsyncClient", _FakeClient),
        patch(
            "scorevision.validator.central.private_track.miners.build_signed_headers",
            return_value={"x-signature": "ok"},
        ),
    ):
        attempt = await send_challenge(_miner(), challenge, hotkey=object(), timeout=5.0)

    assert attempt.response is not None
    assert captured["url"] == "http://127.0.0.1:8000/challenge"
    assert captured["json"]["video_url"] == "https://example.com/snooker.mp4"
    assert captured["json"]["target_frames"] == [50, 150, 250]
