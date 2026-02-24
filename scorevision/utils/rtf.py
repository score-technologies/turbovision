from __future__ import annotations

from logging import getLogger
from typing import Optional

from scorevision.utils.manifest import Manifest

logger = getLogger(__name__)


def calculate_rtf(p95_latency_ms: float, service_rate_fps: float) -> float:
    """
    Compute the Real-Time Factor (RTF) for a miner on a given element.

    RTF = (t_p95_ms / 1000) * (r_e / 5)

    where:
        t_p95_ms      : 95th percentile latency in milliseconds for the batch.
        service_rate_fps (r_e): element service rate in frames per second.

    RTF <= 1.0 means the miner can keep up with real-time at 5 fps.
    """
    if p95_latency_ms < 0:
        raise ValueError("p95_latency_ms must be non-negative")
    if service_rate_fps <= 0:
        raise ValueError("service_rate_fps must be positive")

    return (p95_latency_ms / 10000.0) * (service_rate_fps / 5.0)


def check_rtf_gate(rtf_value: float, threshold: float = 1.0) -> bool:
    """
    Return True if the RTF is within the allowed threshold.

    By default, RTF <= 1.0 passes the gate.
    """
    return rtf_value <= threshold


def get_service_rate_fps_for_element(
    manifest: Manifest,
    element_id: Optional[str],
) -> Optional[float]:
    """
    Look up service_rate_fps for a given element_id in the current manifest.

    Returns:
        float service_rate_fps, or None if not found / misconfigured.

    Logs warnings when lookup fails so operators can detect misconfigurations.
    """
    if manifest is None:
        logger.warning("[RTF] Manifest is None when looking up service_rate_fps.")
        return None

    if not element_id:
        logger.warning("[RTF] No element_id provided for service_rate_fps lookup.")
        return None

    elements = getattr(manifest, "elements", []) or []

    for elem in elements:
        # tolerate both dict-like and object-like elements
        if isinstance(elem, dict):
            eid = elem.get("element_id") or elem.get("id")
            sr = elem.get("service_rate_fps") or elem.get("service_rate")
        else:
            eid = getattr(elem, "element_id", None) or getattr(elem, "id", None)
            sr = getattr(elem, "service_rate_fps", None)
            if sr is None:
                sr = getattr(elem, "service_rate", None)

        if eid != element_id:
            continue

        try:
            return float(sr)
        except (TypeError, ValueError):
            logger.warning(
                "[RTF] service_rate_fps for element_id=%s is not numeric: %r",
                element_id,
                sr,
            )
            return None

    logger.warning(
        "[RTF] Element id '%s' not found in manifest when looking up service_rate_fps.",
        element_id,
    )
    return None
