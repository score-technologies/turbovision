from dataclasses import dataclass
from typing import Any

from numpy import ndarray

from scorevision.chute_template.schemas import SVFrameResult, SVPredictInput


@dataclass
class SVChallenge:
    env: str
    payload: SVPredictInput
    meta: dict[str, Any]
    prompt: str
    challenge_id: str
    frame_numbers: list[int]
    frames: list[ndarray]
    dense_optical_flow_frames: list[ndarray]


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
    smoothness: float
    latency_ms: float
    score: float
    details: dict[str, Any]
