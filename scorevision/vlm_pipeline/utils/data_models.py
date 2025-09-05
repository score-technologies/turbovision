from dataclasses import dataclass

from numpy import ndarray

from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    VLMJudgeFrameResults,
)
from scorevision.utils.settings import get_settings


@dataclass
class PseudoGroundTruth:
    video_name: str
    frame_number: int
    spatial_image: ndarray
    temporal_image: ndarray
    annotation: FrameAnnotation


@dataclass
class AggregatedScore:
    total: float
    breakdown: dict[str, float]


@dataclass
class MinerScore:
    miner_id: str
    score: AggregatedScore
    video_url: str
    frame_numbers: list[int]
    vlm_as_judge_feedback: list[VLMJudgeFrameResults]
    miner_annotations: list[FrameAnnotation]


@dataclass
class Miner:
    id: str
    annotations: list[FrameAnnotation]

    def __post_init__(self):
        settings = get_settings()
        assert (
            len(self.annotations) == settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER
        ), f"Expected {settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER} annotations, got {len(self.annotations)}"
