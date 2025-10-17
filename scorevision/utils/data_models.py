from dataclasses import dataclass, fields, asdict, field
from typing import Any

from numpy import ndarray

from scorevision.chute_template.schemas import SVFrameResult
from scorevision.chute_template.schemas import TVPredictInput
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)


@dataclass
class Evaluation:
    @property
    def average(self) -> float:
        values = [float(getattr(self, f.name)) for f in fields(self)]
        return sum(values) / len(values) if values else 0.0

    def __float__(self) -> float:
        return self.average

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KeypointsScore(Evaluation):
    floor_markings_alignment: float = (
        0.0  # How correct are the keypoint detections based on the alignment of the transformed floor lines?
    )


@dataclass
class ActionScore(Evaluation):
    categorisation: float = (
        0.0  # How correct are the action labels for the scene compared to the Pseudo GT annotations?
    )


@dataclass
class ObjectsScore(Evaluation):
    bbox_placement: float = (
        0.0  # How correct are the objects compared with the PseudoGT annotations (IoU)?
    )
    categorisation: float = (
        0.0  # How correctly are the objects categorised compared with the PseudoGT annotations (i.e. player, ball)?
    )
    team: float = (
        0.0  # How correctly are the teams categorised compared with the PseudoGT annotations?
    )
    enumeration: float = (
        0.0  # How correct are the number of objects detected compared with the PseudoGT annotations?
    )
    tracking_stability: float = (
        0.0  # How stable/smooth are these object detections across the video
    )


@dataclass
class LatencyScore(Evaluation):
    inference: float = (
        0.0  # How quickly does the miner take to produce predictions for the video (1/2**t)
    )


@dataclass
class TotalScore(Evaluation):
    action: ActionScore = field(default_factory=ActionScore)
    keypoints: KeypointsScore = field(default_factory=KeypointsScore)
    objects: ObjectsScore = field(default_factory=ObjectsScore)
    latency: LatencyScore = field(default_factory=LatencyScore)


@dataclass
class SVChallenge:
    env: str
    payload: TVPredictInput
    meta: dict[str, Any]
    prompt: str
    challenge_id: str
    frame_numbers: list[int]
    frames: list[ndarray]
    dense_optical_flow_frames: list[ndarray]
    api_task_id: str | int | None = None
    challenge_type: ChallengeType | None = None


@dataclass
class SVRunOutput:
    success: bool
    latency_ms: float
    predictions: dict[str, list[SVFrameResult]] | None
    error: str | None
    model: str | None = None


@dataclass
class SVPredictResult:
    success: bool
    model: str | None
    latency_seconds: float
    predictions: dict[str, Any] | None
    error: str | None
    raw: dict[str, Any] | None = None


@dataclass
class SVEvaluation:
    acc_breakdown: dict[str, float]
    acc: float
    latency_ms: float
    score: float
    details: dict[str, Any]
