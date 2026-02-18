from pytest import fixture
from pathlib import Path

from scorevision.miner.open_source.chute_template.schemas import TVPredictInput
from scorevision.utils.data_models import SVChallenge
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)
from scorevision.utils.video_processing import FrameStore


@fixture
def fake_frame_store() -> FrameStore:
    return FrameStore(Path("tests/test_data/videos/example_football.mp4"))


@fixture
def fake_payload() -> TVPredictInput:
    return TVPredictInput(
        url="https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4", meta={}
    )


@fixture
def fake_challenge(fake_payload) -> SVChallenge:
    return SVChallenge(
        env="SVEnv",
        payload=fake_payload,
        meta={},
        prompt="ScoreVision video task mock-challenge",
        challenge_id="0",
        frame_numbers=[1, 2, 3],
        frames=[],
        dense_optical_flow_frames=[],
        challenge_type=ChallengeType.FOOTBALL,
    )
