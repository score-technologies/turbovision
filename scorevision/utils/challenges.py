import asyncio
import os
from logging import getLogger
from time import time
from typing import Any
from hashlib import sha256
from json import dumps
from random import randint
from pathlib import Path

from aiohttp import ClientResponseError
from numpy import ndarray

from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import load_hotkey_keypair
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.data_models import SVChallenge
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.video_processing import download_video_cached, FrameStore
from scorevision.utils.image_processing import image_to_base64, pil_from_array
from scorevision.chute_template.schemas import SVFrame
from scorevision.chute_template.schemas import TVPredictInput
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    parse_challenge_type,
    ChallengeType,
    CHALLENGE_ID_LOOKUP,
)
from scorevision.utils.manifest import Manifest

logger = getLogger(__name__)


class ScoreVisionChallengeError(Exception):
    pass


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_challenge_type(challenge: dict) -> ChallengeType:
    ct = parse_challenge_type(challenge.get("challenge_type"))
    if ct:
        return ct
    ct_id = challenge.get("challenge_type_id")
    if isinstance(ct_id, int):
        return CHALLENGE_ID_LOOKUP.get(ct_id, ChallengeType.FOOTBALL)
    if isinstance(ct_id, str) and ct_id.isdigit():
        return CHALLENGE_ID_LOOKUP.get(int(ct_id), ChallengeType.FOOTBALL)
    return ChallengeType.FOOTBALL


def _build_offline_challenge(
    *,
    element_id: str,
    video_url: str,
    manifest_hash: str | None = None,
    window_id: str | None = None,
    task_id: int | None = None,
    challenge_type: str | ChallengeType | None = None,
    fps: int | None = None,
    seed: int | None = None,
) -> dict:
    ct = parse_challenge_type(challenge_type) if challenge_type else None
    ct_value = (ct or ChallengeType.FOOTBALL).value
    offline_task_id = task_id if task_id is not None else 1
    task_id_int = int(offline_task_id)
    payload = {"video_url": video_url}
    if fps is not None:
        payload["fps"] = fps
    return {
        "id": task_id_int,
        "task_id": task_id_int,
        "element_id": element_id,
        "window_id": window_id or "offline",
        "manifest_hash": manifest_hash,
        "challenge_type": ct_value,
        "seed": seed or 0,
        "video_url": video_url,
        "asset_url": video_url,
        "payload": payload,
    }



async def prepare_challenge_payload(
    challenge: dict,
    batch_size: int = 64,
    *,
    video_cache: dict[str, Any] | None = None,
) -> tuple[TVPredictInput, list[int], list[ndarray], list[ndarray], FrameStore]:
    settings = get_settings()

    payload = challenge.get("payload") or {}
    video_url = (
        challenge.get("video_url")
        or challenge.get("asset_url")
        or payload.get("clip_url")
        or payload.get("video_url")
    )
    if not video_url:
        raise ScoreVisionChallengeError("Challenge missing video_url/asset_url/clip_url")

    cached_store: FrameStore | None = None
    cached_path: Path | None = None
    if video_cache is not None:
        cached_store = video_cache.get("store")
        cached_path = video_cache.get("path")

    if cached_store is None:
        _, frame_store = await download_video_cached(
            url=video_url,
            _frame_numbers=[],
            cached_path=cached_path,
        )
        if video_cache is not None:
            video_cache["store"] = frame_store
            video_cache["path"] = frame_store.video_path
    else:
        frame_store = cached_store

    total_frames = frame_store.get_frame_count()
    if total_frames <= 1:
        raise ScoreVisionChallengeError("Could not determine video frame count")

    min_frame = max(1, int(settings.SCOREVISION_VIDEO_MIN_FRAME_NUMBER))
    max_frame_setting = int(settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER)

    max_frame = min(max_frame_setting, total_frames)

    n_select = int(settings.SCOREVISION_VLM_SELECT_N_FRAMES)

    if (max_frame - min_frame) < n_select:
        raise ScoreVisionChallengeError(
            f"Not enough frames to select {n_select} frames "
            f"(min_frame={min_frame}, max_frame={max_frame}, total_frames={total_frames})"
        )
    start = randint(min_frame, max_frame - n_select)
    selected_frame_numbers = list(range(start, start + n_select))
    logger.info(
        f"Selected Frames (dynamic): {selected_frame_numbers} "
        f"(min={min_frame}, max={max_frame}, total={total_frames})"
    )

    select_frames: list[ndarray] = []
    flow_frames: list[ndarray] = []
    for fn in selected_frame_numbers:
        frame = await asyncio.to_thread(frame_store.get_frame, fn)
        select_frames.append(frame)
        flow = await asyncio.to_thread(frame_store.get_flow, fn)
        flow_frames.append(flow)

    logger.info(f"frames {selected_frame_numbers} successful")

    if not select_frames:
        raise ScoreVisionChallengeError("No Frames were successfully extracted from Video")
    if not flow_frames:
        raise ScoreVisionChallengeError("No Dense Optical Flows were successfully computed from Video")

    height, width = select_frames[0].shape[:2]
    meta = {
        "version": 1,
        "width": width or 0,
        "height": height or 0,
        "fps": int(
            challenge.get("fps")
            or payload.get("fps")
            or settings.SCOREVISION_VIDEO_FRAMES_PER_SECOND
        ),
        "task_id": challenge.get("task_id"),
        "challenge_type": challenge.get("challenge_type"),
        "n_frames_total": total_frames,
        "batch_size": batch_size,
        "n_keypoints": 32,  # TODO: update based on challenge type
    }
    if "seed" in challenge:
        meta["seed"] = challenge["seed"]

    payload = TVPredictInput(url=video_url, meta=meta)

    return (
        payload,
        selected_frame_numbers,
        select_frames,
        flow_frames,
        frame_store,
    )


def build_svchallenge_from_parts(
    chal_api: dict,
    payload: TVPredictInput,
    frame_numbers: list[int],
    frames: list[ndarray],
    flows: list[ndarray],
) -> SVChallenge:
    prompt = f"ScoreVision video task {chal_api.get('task_id')}"
    meta = payload.meta | {"seed": chal_api.get("seed", 0)}

    for k in ("element_id", "window_id", "manifest_hash"):
        if chal_api.get(k) is not None:
            meta[k] = chal_api[k]

    canonical = {
        "env": "SVEnv",
        "prompt": prompt,
        "extra": {"meta": meta, "n_frames": len(frames)},
    }
    cid = sha256(
        dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ct = parse_challenge_type(chal_api.get("challenge_type"))
    return SVChallenge(
        env="SVEnv",
        payload=payload,
        meta=meta,
        prompt=prompt,
        challenge_id=cid,
        frame_numbers=frame_numbers,
        frames=frames,
        dense_optical_flow_frames=flows,
        api_task_id=chal_api.get("task_id"),
        challenge_type=ct,
    )


async def get_challenge_from_scorevision_with_source(
    *,
    video_cache: dict[str, Any] | None = None,
    manifest_hash: str | None = None,
    element_id: str | None = None,
) -> tuple[SVChallenge, TVPredictInput, dict, FrameStore]:
    try:
        chal_api = await get_next_challenge_v3(
            manifest_hash=manifest_hash,
            element_id=element_id,
        )
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
    except ScoreVisionChallengeError as e:
        raise e
    except Exception as e:
        raise Exception(f"Unexpected error while fetching challenge: {e}")

    payload, frame_numbers, frames, flows, frame_store = (
        await prepare_challenge_payload(
            challenge=chal_api,
            video_cache=video_cache,
        )
    )
    if not payload:
        raise ScoreVisionChallengeError("Failed to prepare payload from challenge.")

    challenge = build_svchallenge_from_parts(
        chal_api=chal_api,
        payload=payload,
        frame_numbers=frame_numbers,
        frames=frames,
        flows=flows,
    )
    return challenge, payload, chal_api, frame_store

async def complete_task_assignment(
    *,
    challenge_id: int,
    element_id: str | None = None,
) -> None:
    settings = get_settings()
    if not settings.SCOREVISION_API:
        raise ScoreVisionChallengeError("SCOREVISION_API is not set.")

    keypair = load_hotkey_keypair(
        wallet_name=settings.BITTENSOR_WALLET_COLD,
        hotkey_name=settings.BITTENSOR_WALLET_HOT,
    )
    params = build_validator_query_params(keypair)
    if element_id is not None:
        params["element_id"] = element_id

    session = await get_async_client()
    async with session.post(
        f"{settings.SCOREVISION_API}/api/tasks/complete",
        params=params,
        json={"challenge_id": int(challenge_id)},
    ) as response:
        try:
            response.raise_for_status()
        except ClientResponseError as e:
            raise ScoreVisionChallengeError(
                f"HTTP error while completing task assignment: {e}"
            )
        await response.json()

async def get_ground_truth_from_scorevision(
    *,
    challenge_id: int,
    element_id: str | None = None,
) -> Any:
    settings = get_settings()
    if not settings.SCOREVISION_API:
        raise ScoreVisionChallengeError("SCOREVISION_API is not set.")

    keypair = load_hotkey_keypair(
        wallet_name=settings.BITTENSOR_WALLET_COLD,
        hotkey_name=settings.BITTENSOR_WALLET_HOT,
    )
    params = build_validator_query_params(keypair)
    if element_id is not None:
        params["element_id"] = element_id

    session = await get_async_client()
    async with session.get(
        f"{settings.SCOREVISION_API}/api/tasks/{int(challenge_id)}/ground-truth",
        params=params,
    ) as response:
        try:
            response.raise_for_status()
        except ClientResponseError as e:
            if e.status == 403:
                raise ScoreVisionChallengeError(
                    "Assignment not completed for this validator (403)."
                )
            if e.status == 404:
                raise ScoreVisionChallengeError("Ground truth not available (404).")
            raise ScoreVisionChallengeError(f"HTTP error while fetching ground truth: {e}")

        data = await response.json()
        return data.get("ground_truth")

async def get_next_challenge_v3(
    manifest_hash: str | None = None,
    element_id: str | None = None,
) -> dict:
    """
    Challenge client aligned to the current ScoreVision API (/api/tasks/next).
    """
    offline_url = os.getenv("SV_OFFLINE_CHALLENGE_URL")
    if offline_url:
        if element_id is None:
            raise ScoreVisionChallengeError(
                "Offline challenge requires an element_id argument."
            )
        return _build_offline_challenge(
            element_id=element_id,
            video_url=offline_url,
            manifest_hash=manifest_hash,
            window_id=os.getenv("SV_OFFLINE_WINDOW_ID", None),
            task_id=_safe_int(os.getenv("SV_OFFLINE_TASK_ID")),
            challenge_type=os.getenv("SV_OFFLINE_CHALLENGE_TYPE", "football"),
            fps=_safe_int(os.getenv("SV_OFFLINE_FPS")),
            seed=_safe_int(os.getenv("SV_OFFLINE_SEED")) or 0,
        )

    settings = get_settings()

    if not settings.SCOREVISION_API:
        raise ScoreVisionChallengeError("SCOREVISION_API is not set.")

    if element_id is None:
        raise ScoreVisionChallengeError(
            "get_next_challenge_v3() requires an element_id argument."
        )

    keypair = load_hotkey_keypair(
        wallet_name=settings.BITTENSOR_WALLET_COLD,
        hotkey_name=settings.BITTENSOR_WALLET_HOT,
    )
    params = build_validator_query_params(keypair)
    params["element_id"] = element_id

    session = await get_async_client()
    try:
        async with session.get(
            f"{settings.SCOREVISION_API}/api/tasks/next",
            params=params,
        ) as response:
            try:
                response.raise_for_status()
            except ClientResponseError as e:
                if e.status == 404:
                    raise ScoreVisionChallengeError(
                        "No active evaluation window (404 from /api/tasks/next)."
                    )
                if e.status == 409:
                    raise ScoreVisionChallengeError(
                        "Rate limited by /api/tasks/next (409). Back off before retrying."
                    )
                if e.status == 410:
                    raise ScoreVisionChallengeError(
                        "Manifest expired or rejected (410 from /api/tasks/next)."
                    )
                raise

            challenge = await response.json() or None
            if not challenge:
                raise ScoreVisionChallengeError(
                    "Empty challenge payload from /api/challenge/v3."
                )

            if "id" in challenge and "task_id" not in challenge:
                challenge["task_id"] = challenge.pop("id")

            payload = challenge.get("payload") or {}
            if not (
                challenge.get("video_url")
                or challenge.get("asset_url")
                or payload.get("clip_url")
                or payload.get("video_url")
            ):
                raise ScoreVisionChallengeError("Challenge missing video url.")

            ct = _coerce_challenge_type(challenge)
            challenge["challenge_type"] = ct.value

            logger.info(
                "Fetched challenge: task_id=%s element_id=%s",
                challenge.get("task_id"),
                challenge.get("element_id"),
            )
            return challenge
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
