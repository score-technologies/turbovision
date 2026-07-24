from dataclasses import dataclass
from logging import getLogger
import httpx
from scorevision.utils.challenges import get_next_challenge_v3, _coerce_payload_frames
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.schemas import (
    ChallengeFrame,
    CricketDeliveryPrediction,
    FramePrediction,
    TCGGradingPrediction,
)
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass
class Challenge:
    challenge_id: str
    ground_truth: list[FramePrediction] | CricketDeliveryPrediction | TCGGradingPrediction
    groundtruth_type: str = "soccer_action"
    video_url: str | None = None
    image_url: str | None = None
    payload_frames: list[ChallengeFrame] | None = None


def has_sufficient_actions(
    ground_truth: list[FramePrediction] | CricketDeliveryPrediction | TCGGradingPrediction,
    groundtruth_type: str,
) -> bool:
    if groundtruth_type in {"cricket_delivery", "tcg_grading"}:
        return True
    return len(ground_truth) >= get_settings().PRIVATE_MIN_ACTIONS_FOR_CHALLENGE


async def fetch_next_challenge(manifest_hash: str, element_id: str) -> dict:
    return await get_next_challenge_v3(
        manifest_hash=manifest_hash,
        element_id=element_id,
    )


async def fetch_ground_truth(
    challenge_id: str,
    keypair,
    element_id: str | None = None,
    groundtruth_type: str = "soccer_action",
) -> list[FramePrediction] | CricketDeliveryPrediction | TCGGradingPrediction:
    settings = get_settings()
    api_url = settings.PRIVATE_GT_API_URL or settings.SCOREVISION_API
    if not api_url:
        raise RuntimeError("Neither PRIVATE_GT_API_URL nor SCOREVISION_API is configured")

    params = build_validator_query_params(keypair)
    if element_id is not None:
        params["element_id"] = element_id
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{api_url}/api/tasks/{int(challenge_id)}/ground-truth",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    if groundtruth_type == "cricket_delivery":
        raw = data.get("ground_truth", [])
        if not raw:
            raise ValueError("Empty cricket ground truth")
        first = raw[0] or {}
        meta = first.get("meta") if isinstance(first, dict) else None
        if not isinstance(meta, dict):
            raise ValueError("Cricket ground truth missing meta payload")
        return CricketDeliveryPrediction(**meta)

    if groundtruth_type == "tcg_grading":
        raw = data.get("ground_truth")
        if isinstance(raw, list):
            if not raw:
                raise ValueError("Empty TCG grading ground truth")
            raw = raw[0]
        if isinstance(raw, dict) and isinstance(raw.get("meta"), dict):
            meta = raw["meta"]
            if "Header" in meta or "Grading_Features" in meta:
                raw = meta
        if not isinstance(raw, dict):
            raise ValueError("TCG grading ground truth must be an object")
        return TCGGradingPrediction(**raw)

    ground_truth: list[FramePrediction] = []
    for gt in data.get("ground_truth", []):
        action = gt.get("type") or gt.get("action")
        if "frame" not in gt or action is None:
            continue
        ground_truth.append(FramePrediction(frame=gt["frame"], action=action))

    return ground_truth


async def complete_task_assignment(
    challenge_id: str,
    keypair,
    element_id: str | None = None,
) -> None:
    settings = get_settings()
    api_url = settings.PRIVATE_GT_API_URL or settings.SCOREVISION_API
    if not api_url:
        raise RuntimeError("Neither PRIVATE_GT_API_URL nor SCOREVISION_API is configured")

    params = build_validator_query_params(keypair)
    if element_id is not None:
        params["element_id"] = element_id

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{api_url}/api/tasks/complete",
            params=params,
            json={"challenge_id": int(challenge_id)},
        )
        response.raise_for_status()


async def get_challenge_with_ground_truth(
    manifest_hash: str,
    element_id: str,
    keypair,
    groundtruth_type: str = "soccer_action",
    max_retries: int = 3,
) -> Challenge | None:
    for attempt in range(max_retries):
        try:
            chal = await fetch_next_challenge(manifest_hash, element_id)
        except Exception as e:
            logger.error("Failed to fetch challenge (attempt %d/%d): %s", attempt + 1, max_retries, e)
            continue

        payload = chal.get("payload") or {}
        challenge_id = chal.get("task_id") or chal.get("id")
        video_url = (
            chal.get("video_url")
            or (chal.get("asset_url") if groundtruth_type != "tcg_grading" else None)
            or payload.get("video_url")
            or payload.get("clip_url")
        )
        image_url = (
            chal.get("image_url")
            or payload.get("image_url")
            or (chal.get("asset_url") if groundtruth_type == "tcg_grading" else None)
            or (chal.get("url") if groundtruth_type == "tcg_grading" else None)
            or (payload.get("asset_url") if groundtruth_type == "tcg_grading" else None)
            or (payload.get("url") if groundtruth_type == "tcg_grading" else None)
        )
        payload_frames_raw = _coerce_payload_frames(payload)
        payload_frames = [ChallengeFrame(**frame) for frame in payload_frames_raw] or None

        if not challenge_id or (not video_url and not image_url and not payload_frames):
            logger.warning(
                "Challenge missing task_id or challenge asset (video_url/image_url/frames), retrying"
            )
            continue

        try:
            await complete_task_assignment(
                challenge_id=challenge_id,
                keypair=keypair,
                element_id=element_id,
            )
            ground_truth = await fetch_ground_truth(
                challenge_id=challenge_id,
                keypair=keypair,
                element_id=element_id,
                groundtruth_type=groundtruth_type,
            )
        except Exception as e:
            logger.error("Failed to fetch ground truth for %s: %s", challenge_id, e)
            continue

        if not has_sufficient_actions(ground_truth, groundtruth_type=groundtruth_type):
            logger.info(
                "Challenge %s has insufficient actions (%d), retrying",
                challenge_id,
                len(ground_truth) if isinstance(ground_truth, list) else 1,
            )
            continue

        return Challenge(
            challenge_id=str(challenge_id),
            video_url=video_url,
            image_url=image_url,
            payload_frames=payload_frames,
            ground_truth=ground_truth,
            groundtruth_type=groundtruth_type,
        )

    logger.warning("No valid challenge found after %d attempts", max_retries)
    return None
