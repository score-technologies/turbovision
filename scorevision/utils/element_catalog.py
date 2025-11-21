from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

from logging import getLogger

from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import get_subtensor
from scorevision.utils.manifest import get_current_manifest

logger = getLogger(__name__)

WindowScope = Literal["current", "upcoming"]


@dataclass
class ElementInfo:
    element_id: str
    window_id: Optional[str]
    service_rate: Optional[float]
    theta: Optional[float]
    beta: Optional[float]
    telemetry: Dict[str, Any]
    clip_count: Optional[int]


async def _resolve_manifest_for_scope(scope: WindowScope):
    """
    Use get_current_manifest(block_number=...) to resolve the manifest
    for the 'current' or 'upcoming' window.

    - current  : use current chain block.
    - upcoming : current block + SCOREVISION_TEMPO (or 300 by default).
    """
    settings = get_settings()
    st = await get_subtensor()
    block = await st.get_current_block()
    tempo = getattr(settings, "SCOREVISION_TEMPO", 300)

    if scope == "upcoming":
        try:
            block = int(block) + int(tempo)
        except Exception:
            block = int(block) + 300

    logger.info(
        "[ElementCatalog] Resolving manifest for scope=%s at block=%s", scope, block
    )
    manifest = get_current_manifest(block_number=int(block))
    return manifest


def _extract_window_id(manifest: Any) -> Optional[str]:
    for attr in ("window_id", "id"):
        val = getattr(manifest, attr, None)
        if isinstance(val, str) and val:
            return val

    payload = getattr(manifest, "payload", None) or {}
    if isinstance(payload, dict):
        for key in ("window_id", "id"):
            v = payload.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def _extract_elements_raw(manifest: Any) -> List[Any]:
    elems = getattr(manifest, "elements", None)
    if elems is not None:
        return list(elems)

    payload = getattr(manifest, "payload", None) or {}
    elems = payload.get("elements")
    if isinstance(elems, list):
        return elems
    return []


def _extract_element_info(raw: Any, window_id: Optional[str]) -> Optional[ElementInfo]:
    """
    Robust extraction against either dicts or dataclass-like objects.
    """
    element_id = None
    service_rate = None
    theta = None
    beta = None
    telemetry: Dict[str, Any] = {}
    clip_count: Optional[int] = None

    if isinstance(raw, dict):
        element_id = raw.get("element_id") or raw.get("id")
        service_rate = raw.get("service_rate") or raw.get("service_rate_per_block")
        theta = raw.get("theta")
        beta = raw.get("beta")
        telemetry = raw.get("telemetry") or {}
        clip_count = (
            raw.get("clip_count")
            or raw.get("clip_samples")
            or (telemetry.get("clip_count") if isinstance(telemetry, dict) else None)
        )
    else:
        element_id = getattr(raw, "element_id", None) or getattr(raw, "id", None)
        service_rate = getattr(raw, "service_rate", None) or getattr(
            raw, "service_rate_per_block", None
        )
        theta = getattr(raw, "theta", None)
        beta = getattr(raw, "beta", None)
        telemetry = getattr(raw, "telemetry", None) or {}
        clip_count = (
            getattr(raw, "clip_count", None)
            or getattr(raw, "clip_samples", None)
            or (
                getattr(telemetry, "clip_count", None)
                if not isinstance(telemetry, dict)
                else telemetry.get("clip_count")
            )
        )

    if not element_id:
        return None

    def _to_float(val) -> Optional[float]:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    return ElementInfo(
        element_id=str(element_id),
        window_id=window_id,
        service_rate=_to_float(service_rate),
        theta=_to_float(theta),
        beta=_to_float(beta),
        telemetry=telemetry if isinstance(telemetry, dict) else {},
        clip_count=int(clip_count) if clip_count is not None else None,
    )


async def list_elements(window_scope: WindowScope = "current") -> List[ElementInfo]:
    """
    Public API: list elements for 'current' or 'upcoming' windows.

    This is what the CLI uses:
      sv miner elements --window current|upcoming
    """
    manifest = await _resolve_manifest_for_scope(window_scope)
    if manifest is None:
        logger.warning("[ElementCatalog] No manifest for scope=%s", window_scope)
        return []

    window_id = _extract_window_id(manifest)
    raw_elems = _extract_elements_raw(manifest)
    infos: List[ElementInfo] = []

    for raw in raw_elems:
        info = _extract_element_info(raw, window_id)
        if info is not None:
            infos.append(info)

    return infos


async def summarize_window(
    window_scope_or_id: str,
) -> Dict[str, Any]:
    """
    Summarize a window by scope ('current', 'upcoming') or explicit window_id.

    For an explicit window_id:
      - we try 'current' and 'upcoming' manifests and return the one that matches.
      - if none match, we raise a ValueError.
    """
    scope: Optional[WindowScope] = None
    explicit_window_id: Optional[str] = None

    if window_scope_or_id in ("current", "upcoming"):
        scope = window_scope_or_id
    else:
        explicit_window_id = window_scope_or_id

    if scope is not None:
        manifest = await _resolve_manifest_for_scope(scope)
        if manifest is None:
            raise ValueError(f"No manifest found for scope='{scope}'")
        window_id = _extract_window_id(manifest)
        elems = await list_elements(scope)
        return {
            "scope": scope,
            "window_id": window_id,
            "n_elements": len(elems),
            "elements": [asdict(e) for e in elems],
        }

    for candidate_scope in ("current", "upcoming"):
        manifest = await _resolve_manifest_for_scope(candidate_scope)
        if manifest is None:
            continue
        wid = _extract_window_id(manifest)
        if wid == explicit_window_id:
            elems = await list_elements(candidate_scope)
            return {
                "scope": candidate_scope,
                "window_id": wid,
                "n_elements": len(elems),
                "elements": [asdict(e) for e in elems],
            }

    raise ValueError(
        f"Window with id '{explicit_window_id}' not found in current or upcoming manifests."
    )
