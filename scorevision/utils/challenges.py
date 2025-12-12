import asyncio
from logging import getLogger
from time import time
from typing import Any
from hashlib import sha256
from json import dumps
from random import randint
from pathlib import Path
import os

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
)
from scorevision.utils.manifest import Manifest

logger = getLogger(__name__)


class ScoreVisionChallengeError(Exception):
    pass


async def get_challenge_from_scorevision() -> tuple[SVChallenge, TVPredictInput]:
    try:
        chal_api = await get_next_challenge()
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
    except ScoreVisionChallengeError as e:
        raise e
    except Exception as e:
        raise Exception(f"Unexpected error while fetching challenge: {e}")

    payload, frame_numbers, frames, flows, frame_store = (
        await prepare_challenge_payload(challenge=chal_api)
    )
    if not payload:
        if frame_store:
            frame_store.unlink()
        raise ScoreVisionChallengeError("Failed to prepare payload from challenge.")

    # SVChallenge
    prompt = f"ScoreVision video task {chal_api.get('task_id')}"
    meta = payload.meta | {"seed": chal_api.get("seed", 0)}
    canonical = {
        "env": "SVEnv",
        "prompt": prompt,
        "extra": {"meta": meta, "n_frames": len(frames)},
    }

    cid = sha256(
        dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    challenge = SVChallenge(
        env="SVEnv",
        payload=payload,
        meta=meta,
        prompt=prompt,
        challenge_id=cid,
        frame_numbers=frame_numbers,
        frames=frames,
        dense_optical_flow_frames=flows,
    )
    if frame_store:
        frame_store.unlink()
    return challenge, payload


async def prepare_challenge_payload(
    challenge: dict,
    batch_size: int = 64,
    *,
    video_cache: dict[str, Any] | None = None,
) -> tuple[TVPredictInput, list[int], list[ndarray], list[ndarray], FrameStore]:
    settings = get_settings()

    video_url = challenge.get("video_url") or challenge.get("asset_url")
    if not video_url:
        raise ScoreVisionChallengeError("Challenge missing video_url/asset_url")

    frame_numbers = list(
        range(
            settings.SCOREVISION_VIDEO_MIN_FRAME_NUMBER,
            settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER,
        )
    )
    # shuffle(frame_numbers)
    start_frame_number = randint(
        1,
        settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER
        - settings.SCOREVISION_VLM_SELECT_N_FRAMES
        - 1,
    )
    selected_frame_numbers = frame_numbers[
        start_frame_number : start_frame_number
        + settings.SCOREVISION_VLM_SELECT_N_FRAMES
    ]
    logger.info(f"Selected Frames for Testing: {selected_frame_numbers}")

    cached_store: FrameStore | None = None
    cached_path: Path | None = None
    if video_cache is not None:
        cached_store = video_cache.get("store")
        cached_path = video_cache.get("path")

    if cached_store is None:
        video_name, frame_store = await download_video_cached(
            url=video_url,
            _frame_numbers=selected_frame_numbers,
            cached_path=cached_path,
        )
        if video_cache is not None:
            video_cache["store"] = frame_store
            video_cache["path"] = frame_store.video_path
    else:
        frame_store = cached_store

    select_frames: list[ndarray] = []
    flow_frames: list[ndarray] = []
    for fn in selected_frame_numbers:
        frame = await asyncio.to_thread(frame_store.get_frame, fn)
        select_frames.append(frame)
        flow = await asyncio.to_thread(frame_store.get_flow, fn)
        flow_frames.append(flow)

    logger.info(f"frames {selected_frame_numbers} successful")
    if not select_frames:
        raise ScoreVisionChallengeError(
            "No Frames were successfully extracted from Video"
        )
    if not flow_frames:
        raise ScoreVisionChallengeError(
            "No Dense Optical Flows were successfully computed from Video"
        )

    height, width = select_frames[0].shape[:2]
    meta = {
        "version": 1,
        "width": width or 0,
        "height": height or 0,
        "fps": int(
            challenge.get("fps") or settings.SCOREVISION_VIDEO_FRAMES_PER_SECOND
        ),
        "task_id": challenge.get("task_id"),
        "challenge_type": challenge.get("challenge_type"),
    }
    if "seed" in challenge:
        meta["seed"] = challenge["seed"]
    meta["batch_size"] = batch_size
    meta["n_keypoints"] = (
        32  # TODO: update n_keypoints based on challenge type (32 is for football)
    )
    payload = TVPredictInput(url=video_url, meta=meta)
    return (
        payload,
        selected_frame_numbers,
        select_frames,
        flow_frames,
        frame_store,
    )


async def get_next_challenge() -> dict:
    """
    Fetches the next video challenge from ScoreVision API (legacy v2).
    """
    settings = get_settings()

    if not settings.SCOREVISION_API:
        raise ScoreVisionChallengeError("SCOREVISION_API is not set.")

    keypair = load_hotkey_keypair(
        wallet_name=settings.BITTENSOR_WALLET_COLD,
        hotkey_name=settings.BITTENSOR_WALLET_HOT,
    )

    # Build query parameters required to authenticate with the validator API
    params = build_validator_query_params(keypair)

    session = await get_async_client()
    async with session.get(
        f"{settings.SCOREVISION_API}/api/tasks/next/v2", params=params
    ) as response:
        response.raise_for_status()
        challenge = await response.json() or None
        if not challenge:
            raise ScoreVisionChallengeError("No challenge available from API")

        if "id" in challenge and "task_id" not in challenge:
            challenge["task_id"] = challenge.pop("id")

        if not (challenge.get("video_url") or challenge.get("asset_url")):
            raise ScoreVisionChallengeError("Challenge missing video url.")

        ct = (
            parse_challenge_type(challenge.get("challenge_type"))
            or ChallengeType.FOOTBALL
        )
        challenge["challenge_type"] = ct.value

        logger.info(f"Fetched challenge: task_id={challenge.get('task_id')}")
        return challenge


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
    use_v3 = os.getenv("SCOREVISION_USE_CHALLENGE_V3", "0") not in (
        "0",
        "false",
        "False",
    )

    try:
        if use_v3:
            if manifest_hash is None:
                raise ScoreVisionChallengeError(
                    "SCOREVISION_USE_CHALLENGE_V3=1 but no manifest_hash provided "
                    "to get_challenge_from_scorevision_with_source()."
                )
            chal_api = await get_next_challenge_v3(
                manifest_hash=manifest_hash,
                element_id=element_id,
            )
        else:
            chal_api = await get_next_challenge()
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


async def get_next_challenge_v3(
    manifest_hash: str | None = None,
    element_id: str | None = None,
) -> dict:
    """
    New v3 challenge client using Manifest hash + optional element_id.

    Request:
      GET /api/challenge/v3
        - query params: auth + manifest_hash + optional element_id
        - headers:
            X-Manifest-Hash: <manifest_hash>
            X-Element-Id: <element_id> (si fourni)

    Response attendu (exemple simplifié):
      {
        "task_id": "...",
        "video_url": "...",
        "fps": 25,
        "seed": 123,
        "element_id": "PlayerDetect_v1@1.0",
        "window_id": "block-123456",
        "manifest_hash": "...",
        ...
      }
    """
    settings = get_settings()

    if not settings.SCOREVISION_API:
        raise ScoreVisionChallengeError("SCOREVISION_API is not set.")

    if manifest_hash is None:
        raise ScoreVisionChallengeError(
            "get_next_challenge_v3() requires a manifest_hash argument."
        )

    keypair = load_hotkey_keypair(
        wallet_name=settings.BITTENSOR_WALLET_COLD,
        hotkey_name=settings.BITTENSOR_WALLET_HOT,
    )
    params = build_validator_query_params(keypair)
    params["manifest_hash"] = manifest_hash
    if element_id is not None:
        params["element_id"] = element_id

    headers: dict[str, str] = {
        "X-Manifest-Hash": manifest_hash,
    }
    if element_id is not None:
        headers["X-Element-Id"] = element_id

    session = await get_async_client()
    try:
        async with session.get(
            f"{settings.SCOREVISION_API}/api/challenge/v3",
            params=params,
            headers=headers,
        ) as response:
            try:
                response.raise_for_status()
            except ClientResponseError as e:
                if e.status == 404:
                    raise ScoreVisionChallengeError(
                        "No active evaluation window (404 from /api/challenge/v3)."
                    )
                if e.status == 409:
                    raise ScoreVisionChallengeError(
                        "Rate limited by /api/challenge/v3 (409). Back off before retrying."
                    )
                if e.status == 410:
                    raise ScoreVisionChallengeError(
                        "Manifest expired or rejected (410 from /api/challenge/v3)."
                    )
                raise

            challenge = await response.json() or None
            if not challenge:
                raise ScoreVisionChallengeError(
                    "Empty challenge payload from /api/challenge/v3."
                )

            if "id" in challenge and "task_id" not in challenge:
                challenge["task_id"] = challenge.pop("id")

            if not (challenge.get("video_url") or challenge.get("asset_url")):
                raise ScoreVisionChallengeError("Challenge missing video url.")

            ct = (
                parse_challenge_type(challenge.get("challenge_type"))
                or ChallengeType.FOOTBALL
            )
            challenge["challenge_type"] = ct.value

            resp_mh = challenge.get("manifest_hash")
            if resp_mh is not None and resp_mh != manifest_hash:
                raise ScoreVisionChallengeError(
                    f"Manifest hash mismatch in /api/challenge/v3 response "
                    f"(sent={manifest_hash}, got={resp_mh})."
                )

            if not challenge.get("element_id"):
                raise ScoreVisionChallengeError(
                    "Missing element_id in /api/challenge/v3 response."
                )
            if not challenge.get("window_id"):
                raise ScoreVisionChallengeError(
                    "Missing window_id in /api/challenge/v3 response."
                )

            if element_id is not None and str(challenge.get("element_id")) != str(
                element_id
            ):
                raise ScoreVisionChallengeError(
                    f"Element_id mismatch in /api/challenge/v3 response "
                    f"(requested={element_id}, got={challenge.get('element_id')})."
                )

            logger.info(
                "Fetched v3 challenge: task_id=%s element_id=%s window_id=%s",
                challenge.get("task_id"),
                challenge.get("element_id"),
                challenge.get("window_id"),
            )
            return challenge
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching v3 challenge: {e}")