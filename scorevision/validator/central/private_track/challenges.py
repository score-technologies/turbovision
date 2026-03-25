from dataclasses import dataclass
from logging import getLogger
import httpx
from scorevision.utils.challenges import get_next_challenge_v3
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass
class Challenge:
    challenge_id: str
    video_url: str
    ground_truth: list[FramePrediction]


def has_sufficient_actions(ground_truth: list) -> bool:
    return len(ground_truth) >= get_settings().PRIVATE_MIN_ACTIONS_FOR_CHALLENGE


async def fetch_next_challenge(manifest_hash: str, element_id: str) -> dict:
    return await get_next_challenge_v3(
        manifest_hash=manifest_hash,
        element_id=element_id,
    )


async def fetch_ground_truth(challenge_id: str, keypair, element_id: str | None = None) -> list[FramePrediction]:
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

    return [
        FramePrediction(frame=gt["frame"], action=gt["type"])
        for gt in data.get("ground_truth", [])
    ]


async def get_challenge_with_ground_truth(
    manifest_hash: str,
    element_id: str,
    keypair,
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
            or chal.get("asset_url")
            or payload.get("video_url")
            or payload.get("clip_url")
        )

        if not challenge_id or not video_url:
            logger.warning("Challenge missing task_id or video_url, retrying")
            continue

        try:
            ground_truth = await fetch_ground_truth(
                challenge_id=challenge_id,
                keypair=keypair,
                element_id=element_id,
            )
        except Exception as e:
            logger.error("Failed to fetch ground truth for %s: %s", challenge_id, e)
            continue

        if not has_sufficient_actions(ground_truth):
            logger.info(
                "Challenge %s has insufficient actions (%d), retrying",
                challenge_id,
                len(ground_truth),
            )
            continue

        return Challenge(
            challenge_id=str(challenge_id),
            video_url=video_url,
            ground_truth=ground_truth,
        )

    logger.warning("No valid challenge found after %d attempts", max_retries)
    return None
