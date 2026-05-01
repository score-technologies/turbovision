from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LocalRunResult:
    success: bool
    predictions: dict[str, Any] | None
    latency_ms: float
    error: str | None = None


def run_local_inference_from_hf(
    *,
    model_repo: str,
    revision: str,
    payload_frames: list[dict[str, Any]],
    n_keypoints: int = 32,
    max_repo_bytes: int = 30 * 1024 * 1024,
    memory_bytes: int = 8 * 1024 * 1024 * 1024,
    cpu_seconds: int = 30,
    wall_timeout_seconds: int = 45,
) -> LocalRunResult:
    # Placeholder stub to keep imports/runtime wiring stable.
    _ = (
        model_repo,
        revision,
        payload_frames,
        n_keypoints,
        max_repo_bytes,
        memory_bytes,
        cpu_seconds,
        wall_timeout_seconds,
    )
    return LocalRunResult(success=True, predictions={"frames": []}, latency_ms=0.0, error=None)
