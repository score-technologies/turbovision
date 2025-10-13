from logging import getLogger
from time import time
from typing import Any
from hashlib import sha256
from json import dumps
from random import shuffle, randint

from aiohttp import ClientResponseError
from numpy import ndarray

from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import load_hotkey_keypair
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.data_models import SVChallenge
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.video_processing import download_video
from scorevision.utils.image_processing import image_to_base64, pil_from_array
from scorevision.chute_template.schemas import SVPredictInput, SVFrame
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    parse_challenge_type,
    ChallengeType,
)

logger = getLogger(__name__)


class ScoreVisionChallengeError(Exception):
    pass


async def get_challenge_from_scorevision() -> tuple[SVChallenge, SVPredictInput]:
    try:
        chal_api = await get_next_challenge()
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
    except ScoreVisionChallengeError as e:
        raise e
    except Exception as e:
        raise Exception(f"Unexpected error while fetching challenge: {e}")

    payload, frame_numbers, frames, flows = await prepare_challenge_payload(
        challenge=chal_api
    )
    if not payload:
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
    return challenge, payload


async def prepare_challenge_payload(
    challenge: dict, batch_size: int | None = 64, mock_after_n_frames: int | None = None
) -> tuple[SVPredictInput, list[int], list[ndarray], list[ndarray], dict[int, ndarray]]:
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

    video_name, frames, flows = await download_video(
        url=video_url, frame_numbers=selected_frame_numbers
    )
    selected_frame_numbers = list(flows.keys())
    logger.info(f"frames {selected_frame_numbers} successful")
    if not any(frames):
        raise ScoreVisionChallengeError(
            "No Frames were successfully extracted from Video"
        )
    if not any(flows):
        raise ScoreVisionChallengeError(
            "No Dense Optical Flows were successfully computed from Video"
        )
    height, width = frames[0].shape[:2]
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
    if batch_size:
        meta["batch_size"] = batch_size
    if mock_after_n_frames:
        meta["mock_after_n_frames"] = mock_after_n_frames
    select_frames = [frames[frame_number] for frame_number in selected_frame_numbers]
    payload = SVPredictInput(url=video_url, meta=meta)
    return (
        payload,
        selected_frame_numbers,
        select_frames,
        list(flows.values()),
        frames,
    )


async def get_next_challenge() -> dict:
    """
    Fetches the next video challenge from ScoreVision API.
    Returns a dict like:
      {
        "task_id": "...",     # we will propagate this end-to-end
        "video_url": "...",   # or "asset_url"
        "fps": 25|30,         # optional (fallback 30)
        "seed": <int>,        # optional
        ...
      }
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
    payload: SVPredictInput,
    frame_numbers: list[int],
    frames: list[ndarray],
    flows: list[ndarray],
) -> SVChallenge:
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


async def get_challenge_from_scorevision_with_source() -> (
    tuple[SVChallenge, SVPredictInput, dict, dict]
):
    try:
        chal_api = await get_next_challenge()
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
    except ScoreVisionChallengeError as e:
        raise e
    except Exception as e:
        raise Exception(f"Unexpected error while fetching challenge: {e}")

    payload, frame_numbers, frames, flows, all_frames = await prepare_challenge_payload(
        challenge=chal_api
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
    return challenge, payload, chal_api, all_frames
