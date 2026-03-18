import asyncio
import os
from logging import getLogger
from time import time
from typing import Any
from hashlib import sha256
from json import dumps
from random import randint
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from aiohttp import ClientResponseError
from base64 import b64decode
from numpy import ndarray, frombuffer, uint8, zeros
from cv2 import imdecode, IMREAD_COLOR
from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import load_hotkey_keypair
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.data_models import SVChallenge
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.video_processing import (
    download_video_cached,
    FrameStore,
    InMemoryFrameStore,
)
from scorevision.utils.image_processing import image_to_b64string
from scorevision.chute_template.schemas import TVFrame, TVPredictInput
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation
from scorevision.utils.manifest import Manifest

logger = getLogger(__name__)


class ScoreVisionChallengeError(Exception):
    pass


def _parse_ground_truth_payload(
    ground_truth: Any, challenge_id: int
) -> list[PseudoGroundTruth]:
    if isinstance(ground_truth, list) and all(
        isinstance(item, PseudoGroundTruth) for item in ground_truth
    ):
        return ground_truth

    if not isinstance(ground_truth, dict):
        return []

    annotations = ground_truth.get("annotations")
    if not isinstance(annotations, list):
        return []

    grouped: dict[int, list[BoundingBox]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        except (TypeError, ValueError):
            continue

        label = ann.get("class") or ann.get("label")
        if label is None:
            label = ""

        frame_idx = ann.get("frame_idx")
        if isinstance(frame_idx, str) and frame_idx.isdigit():
            frame_idx = int(frame_idx)
        if not isinstance(frame_idx, int):
            frame_idx = 0

        grouped.setdefault(frame_idx, []).append(
            BoundingBox(
                bbox_2d=(x1, y1, x2, y2),
                label=str(label),
                cluster_id=None,
            )
        )

    if not grouped:
        return []

    spatial_stub = zeros((1, 1, 3), dtype=uint8)
    temporal_stub = zeros((1, 1, 3), dtype=uint8)
    pseudo_gt: list[PseudoGroundTruth] = []
    for frame_number in sorted(grouped.keys()):
        pseudo_gt.append(
            PseudoGroundTruth(
                video_name=str(challenge_id),
                frame_number=frame_number,
                spatial_image=spatial_stub,
                temporal_image=temporal_stub,
                annotation=FrameAnnotation(
                    bboxes=grouped[frame_number],
                    category=Action.NONE,
                    confidence=100,
                    reason="ScoreVision API ground truth",
                ),
            )
        )
    return pseudo_gt


def _looks_like_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def _normalize_challenge_asset_url(url: str) -> str:
    """
    Normalize challenge object URLs to a stable public endpoint.

    Some challenge payloads contain pre-signed R2 URLs that can expire before processing.
    For known challenge-object paths we rewrite to the public manako host.
    """
    try:
        parsed = urlparse(url)
        marker = "/challenge-objects/"
        if marker not in parsed.path:
            return url

        query = parse_qs(parsed.query, keep_blank_values=True)
        is_presigned = "X-Amz-Algorithm" in query or "X-Amz-Signature" in query
        is_r2_storage_host = parsed.netloc.endswith(".r2.cloudflarestorage.com")
        if not (is_presigned or is_r2_storage_host):
            return url

        suffix = parsed.path[parsed.path.index(marker) :]
        rewritten = f"https://manako.scoredata.me{suffix}"
        if rewritten != url:
            logger.info("Rewrote challenge asset URL to public endpoint: %s", rewritten)
        return rewritten
    except Exception:
        return url


def _normalize_challenge_payload_urls(challenge: dict) -> None:
    if not isinstance(challenge, dict):
        return

    for key in ("video_url", "asset_url"):
        value = challenge.get(key)
        if isinstance(value, str) and value:
            challenge[key] = _normalize_challenge_asset_url(value)

    payload = challenge.get("payload")
    if not isinstance(payload, dict):
        return

    for key in ("video_url", "clip_url", "asset_url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            payload[key] = _normalize_challenge_asset_url(value)

    frames = payload.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            value = frame.get("url")
            if isinstance(value, str) and value:
                frame["url"] = _normalize_challenge_asset_url(value)


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_challenge_type_id(challenge: dict) -> int | None:
    ct_id = challenge.get("challenge_type_id")
    if isinstance(ct_id, int) and not isinstance(ct_id, bool):
        return ct_id
    if isinstance(ct_id, str):
        s = ct_id.strip()
        if s.isdigit():
            return int(s)

    # Backward compatibility for legacy API payloads that still return strings.
    ct = challenge.get("challenge_type")
    if isinstance(ct, int) and not isinstance(ct, bool):
        return ct
    if isinstance(ct, str):
        s = ct.strip()
        if s.isdigit():
            return int(s)
        return {
            "football": 0,
            "soccer": 0,
            "cricket": 1,
            "basketball": 2,
        }.get(s.lower())
    return None


def _coerce_payload_frames(payload: dict) -> list[dict[str, object]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    result: list[dict[str, object]] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id") or item.get("frameid") or item.get("id")
        url = item.get("url")
        if isinstance(url, str) and url:
            url = _normalize_challenge_asset_url(url)
        data = item.get("data")
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if isinstance(frame_id, int) and (isinstance(url, str) or isinstance(data, str)):
            result.append({"frame_id": frame_id, "url": url, "data": data})
    return result


def _decode_frame_bytes(data: bytes) -> ndarray:
    arr = frombuffer(data, dtype=uint8)
    img = imdecode(arr, IMREAD_COLOR)
    if img is None:
        raise ScoreVisionChallengeError("Failed to decode frame image data")
    return img


async def _download_frame_from_url(url: str) -> ndarray:
    url = _normalize_challenge_asset_url(url)
    session = await get_async_client()
    async with session.get(url) as response:
        if response.status != 200:
            txt = await response.text()
            raise ScoreVisionChallengeError(
                f"Frame download failed {response.status}: {txt[:200]}"
            )
        data = await response.read()

    return _decode_frame_bytes(data)


async def _load_payload_frames(frame_entries: list[dict[str, object]]) -> list[ndarray]:
    tasks = []
    for entry in frame_entries:
        data = entry.get("data")
        url = entry.get("url")
        if isinstance(data, str) and data:
            try:
                decoded = b64decode(data)
            except Exception as e:
                raise ScoreVisionChallengeError(
                    f"Failed to decode base64 frame data: {e}"
                )
            tasks.append(asyncio.to_thread(_decode_frame_bytes, decoded))
        elif isinstance(url, str) and url:
            tasks.append(asyncio.create_task(_download_frame_from_url(url)))
        else:
            raise ScoreVisionChallengeError("Frame entry missing url or data")
    frames = await asyncio.gather(*tasks)
    if not frames:
        raise ScoreVisionChallengeError("No frames were downloaded from payload")
    return frames


def _build_offline_challenge(
    *,
    element_id: str,
    video_url: str,
    manifest_hash: str | None = None,
    window_id: str | None = None,
    task_id: int | None = None,
    challenge_type_id: int | None = None,
    fps: int | None = None,
    seed: int | None = None,
) -> dict:
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
        "challenge_type_id": challenge_type_id if challenge_type_id is not None else 0,
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
    frame_numbers: list[int] | None = None,
) -> tuple[TVPredictInput, list[int], list[ndarray], list[ndarray], FrameStore | InMemoryFrameStore]:
    settings = get_settings()
    _normalize_challenge_payload_urls(challenge)

    payload = challenge.get("payload") or {}
    payload_frames = _coerce_payload_frames(payload)

    if payload_frames:
        payload_frames_sorted = sorted(
            payload_frames, key=lambda entry: int(entry["frame_id"])
        )
        all_frames = await _load_payload_frames(payload_frames_sorted)
        all_frame_numbers = [
            int(entry["frame_id"]) for entry in payload_frames_sorted
        ]
        frame_urls_by_id: dict[int, str] = {}
        for entry in payload_frames_sorted:
            fid = int(entry["frame_id"])
            url = entry.get("url")
            if isinstance(url, str) and url:
                frame_urls_by_id[fid] = url

        if len(set(all_frame_numbers)) != len(all_frame_numbers):
            raise ScoreVisionChallengeError("Duplicate frame_id values in payload.frames")

        frame_map = {
            fid: frame for fid, frame in zip(all_frame_numbers, all_frames, strict=True)
        }
        frame_store = InMemoryFrameStore(frame_map)

        total_frames = len(all_frame_numbers)

        if frame_numbers is not None:
            selected_frame_numbers = [fn for fn in frame_numbers if fn in frame_map]
            n_select = len(selected_frame_numbers)
            logger.info(
                "Selected Payload Frames (explicit): %s (total=%s)",
                selected_frame_numbers,
                total_frames,
            )
        else:
            n_select_cfg = int(settings.SCOREVISION_VLM_SELECT_N_FRAMES)
            n_select = min(n_select_cfg, total_frames)
            if n_select <= 0:
                raise ScoreVisionChallengeError(
                    "SCOREVISION_VLM_SELECT_N_FRAMES must be positive"
                )

            start_idx = randint(0, total_frames - n_select)
            selected_frame_numbers = all_frame_numbers[start_idx : start_idx + n_select]
            logger.info(
                "Selected Payload Frames (dynamic): %s (total=%s, n_select=%s)",
                selected_frame_numbers,
                total_frames,
                n_select,
            )

        select_frames: list[ndarray] = []
        flow_frames: list[ndarray] = []
        for fid in selected_frame_numbers:
            frame = await asyncio.to_thread(frame_store.get_frame, fid)
            select_frames.append(frame)
            flow = await asyncio.to_thread(frame_store.get_flow, fid)
            flow_frames.append(flow)

        if not select_frames:
            raise ScoreVisionChallengeError("No frames were successfully loaded from payload")

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
            "challenge_type_id": challenge.get("challenge_type_id"),
            "n_frames_total": total_frames,
            "batch_size": batch_size,
            "n_keypoints": 32,  # TODO: update based on challenge type
            "min_frames_required": n_select,
            "frames_source": "payload",
        }
        if "seed" in challenge:
            meta["seed"] = challenge["seed"]

        payload_out_frames: list[TVFrame] = []
        for fid, frame in zip(all_frame_numbers, all_frames, strict=True):
            b64 = image_to_b64string(frame)
            if not b64:
                raise ScoreVisionChallengeError("Failed to encode frame image data")
            payload_out_frames.append(
                TVFrame(
                    frame_id=fid,
                    url=frame_urls_by_id.get(fid),
                    data=b64,
                )
            )

        payload_out = TVPredictInput(
            url=None,
            frames=payload_out_frames,
            meta=meta,
        )

        return (
            payload_out,
            selected_frame_numbers,
            select_frames,
            flow_frames,
            frame_store,
        )

    video_url = (
        challenge.get("video_url")
        or challenge.get("asset_url")
        or payload.get("clip_url")
        or payload.get("video_url")
    )
    if not video_url:
        raise ScoreVisionChallengeError("Challenge missing video_url/asset_url/clip_url")
    video_url = _normalize_challenge_asset_url(video_url)

    if _looks_like_image_url(video_url):
        logger.info("Detected image challenge URL, loading as single-frame payload: %s", video_url)
        frame = await _download_frame_from_url(video_url)
        frame_store = InMemoryFrameStore({0: frame})
        total_frames = 1

        if frame_numbers is not None:
            selected_frame_numbers = [fn for fn in frame_numbers if fn == 0]
            if not selected_frame_numbers:
                selected_frame_numbers = [0]
            n_select = len(selected_frame_numbers)
            logger.info(
                "Selected Image Frame (explicit): %s (total=%s)",
                selected_frame_numbers,
                total_frames,
            )
        else:
            selected_frame_numbers = [0]
            n_select = 1
            logger.info("Selected Image Frame (dynamic): %s (total=%s)", selected_frame_numbers, total_frames)

        select_frames: list[ndarray] = []
        flow_frames: list[ndarray] = []
        for fn in selected_frame_numbers:
            image = await asyncio.to_thread(frame_store.get_frame, fn)
            select_frames.append(image)
            flow = await asyncio.to_thread(frame_store.get_flow, fn)
            flow_frames.append(flow)

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
            "challenge_type_id": challenge.get("challenge_type_id"),
            "n_frames_total": total_frames,
            "batch_size": batch_size,
            "n_keypoints": 32,  # TODO: update based on challenge type
            "min_frames_required": n_select,
            "frames_source": "image_url",
        }
        if "seed" in challenge:
            meta["seed"] = challenge["seed"]

        b64 = image_to_b64string(select_frames[0])
        if not b64:
            raise ScoreVisionChallengeError("Failed to encode image challenge frame data")
        payload_out = TVPredictInput(
            url=None,
            frames=[TVFrame(frame_id=0, url=video_url, data=b64)],
            meta=meta,
        )

        return (
            payload_out,
            selected_frame_numbers,
            select_frames,
            flow_frames,
            frame_store,
        )

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

    if frame_numbers is not None:
        selected_frame_numbers = [fn for fn in frame_numbers if 0 <= fn < total_frames]
        logger.info(
            f"Selected Frames (explicit): {selected_frame_numbers} "
            f"(total={total_frames})"
        )
    else:
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
        "challenge_type_id": challenge.get("challenge_type_id"),
        "n_frames_total": total_frames,
        "batch_size": batch_size,
        "n_keypoints": 32,  # TODO: update based on challenge type
    }
    if "seed" in challenge:
        meta["seed"] = challenge["seed"]

    payload = TVPredictInput(url=video_url, frames=None, meta=meta)

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
    ct_id = _coerce_challenge_type_id(chal_api)
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
        challenge_type_id=ct_id,
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
        logger.info(
            "[GroundTruth] task_id=%s element_id=%s full_response=%r",
            challenge_id,
            element_id,
            data,
        )
        ground_truth = data.get("ground_truth")
        parsed = _parse_ground_truth_payload(ground_truth, int(challenge_id))
        if parsed:
            logger.info(
                "[GroundTruth] parsed %s pseudo_gt frame(s) from API payload",
                len(parsed),
            )
            return parsed
        logger.warning(
            "[GroundTruth] could not parse API ground_truth payload into pseudo_gt, returning raw payload"
        )
        return ground_truth

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
        offline_challenge_type_id = _safe_int(os.getenv("SV_OFFLINE_CHALLENGE_TYPE_ID"))
        if offline_challenge_type_id is None:
            offline_challenge_type_id = _coerce_challenge_type_id(
                {"challenge_type": os.getenv("SV_OFFLINE_CHALLENGE_TYPE")}
            )
        return _build_offline_challenge(
            element_id=element_id,
            video_url=offline_url,
            manifest_hash=manifest_hash,
            window_id=os.getenv("SV_OFFLINE_WINDOW_ID", None),
            task_id=_safe_int(os.getenv("SV_OFFLINE_TASK_ID")),
            challenge_type_id=offline_challenge_type_id,
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
            _normalize_challenge_payload_urls(challenge)
            logger.info("Challenge API raw response: %s", dumps(challenge, default=str))

            if "id" in challenge and "task_id" not in challenge:
                challenge["task_id"] = challenge.pop("id")

            payload = challenge.get("payload") or {}
            has_video_url = (
                challenge.get("video_url")
                or challenge.get("asset_url")
                or payload.get("clip_url")
                or payload.get("video_url")
            )
            has_payload_frames = bool(_coerce_payload_frames(payload))
            if not (has_video_url or has_payload_frames):
                raise ScoreVisionChallengeError("Challenge missing video url or payload frames.")

            challenge["challenge_type_id"] = _coerce_challenge_type_id(challenge)

            logger.info(
                "Fetched challenge: task_id=%s element_id=%s",
                challenge.get("task_id"),
                challenge.get("element_id"),
            )
            return challenge
    except ClientResponseError as e:
        raise ScoreVisionChallengeError(f"HTTP error while fetching challenge: {e}")
