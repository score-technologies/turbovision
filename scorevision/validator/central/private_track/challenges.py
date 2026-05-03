from dataclasses import dataclass
from logging import getLogger
from urllib.parse import parse_qs, urlparse
import httpx
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.schemas import ChallengeFrame, CricketDeliveryPrediction, FramePrediction
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

_CRICKET_GT_FIELDS = {
    "match",
    "matchid",
    "inningsid",
    "innings",
    "overid",
    "over",
    "ball_in_over",
    "ball",
    "ballid",
    "xlsx_overs",
    "scorecard_overs",
    "overs",
    "kph",
    "release_y",
    "rel_y",
    "release_z",
    "rel_z",
    "bounce_x",
    "bounce_y",
    "impact_x",
    "impact_y",
    "impact_z",
    "interception_distance",
    "inter_d",
    "stump_y",
    "stump_z",
    "swing_angle",
    "swing_deg",
    "deviation",
    "deviation_deg",
    "runs",
    "wickets",
    "wkts",
}


@dataclass
class Challenge:
    challenge_id: str
    ground_truth: list[FramePrediction] | CricketDeliveryPrediction
    video_url: str | None = None
    payload_frames: list[ChallengeFrame] | None = None


def _is_cricket_ground_truth_dict(payload: dict) -> bool:
    return bool(_CRICKET_GT_FIELDS.intersection(payload.keys()))


def _normalize_challenge_asset_url(url: str) -> str:
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
        return f"https://manako.scoredata.me{suffix}"
    except Exception:
        return url


def _coerce_payload_frames(payload: dict) -> list[dict[str, object]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []

    result: list[dict[str, object]] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id")
        if frame_id is None:
            frame_id = item.get("frameid")
        if frame_id is None:
            frame_id = item.get("id")
        url = item.get("url")
        if isinstance(url, str) and url:
            url = _normalize_challenge_asset_url(url)
        data = item.get("data")
        if isinstance(frame_id, str) and frame_id.isdigit():
            frame_id = int(frame_id)
        if isinstance(frame_id, int) and (isinstance(url, str) or isinstance(data, str)):
            result.append({"frame_id": frame_id, "url": url, "data": data})
    return result


def parse_ground_truth_payload(payload: object) -> list[FramePrediction] | CricketDeliveryPrediction:
    if isinstance(payload, dict):
        if _is_cricket_ground_truth_dict(payload):
            return CricketDeliveryPrediction(**payload)
        return []

    if not isinstance(payload, list) or not payload:
        return []

    first_item = payload[0]
    if len(payload) == 1 and isinstance(first_item, dict) and _is_cricket_ground_truth_dict(first_item):
        return CricketDeliveryPrediction(**first_item)

    ground_truth: list[FramePrediction] = []
    for gt in payload:
        if not isinstance(gt, dict):
            continue
        action = gt.get("type") or gt.get("action")
        if "frame" not in gt or action is None:
            continue
        ground_truth.append(FramePrediction(frame=gt["frame"], action=action))
    return ground_truth


def has_sufficient_actions(ground_truth: list[FramePrediction] | CricketDeliveryPrediction) -> bool:
    if isinstance(ground_truth, CricketDeliveryPrediction):
        return True
    return len(ground_truth) >= get_settings().PRIVATE_MIN_ACTIONS_FOR_CHALLENGE


async def fetch_next_challenge(manifest_hash: str, element_id: str) -> dict:
    from scorevision.utils.challenges import get_next_challenge_v3

    return await get_next_challenge_v3(
        manifest_hash=manifest_hash,
        element_id=element_id,
    )


async def fetch_ground_truth(
    challenge_id: str,
    keypair,
    element_id: str | None = None,
) -> list[FramePrediction] | CricketDeliveryPrediction:
    settings = get_settings()
    api_url = settings.PRIVATE_GT_API_URL or settings.SCOREVISION_API
    if not api_url:
        raise RuntimeError("Neither PRIVATE_GT_API_URL nor SCOREVISION_API is configured")

    params = build_validator_query_params(keypair) if keypair is not None else {}
    if element_id is not None:
        params["element_id"] = element_id
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{api_url}/api/tasks/{int(challenge_id)}/ground-truth",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    return parse_ground_truth_payload(data.get("ground_truth", []))


async def complete_task_assignment(
    challenge_id: str,
    keypair,
    element_id: str | None = None,
) -> None:
    settings = get_settings()
    api_url = settings.PRIVATE_GT_API_URL or settings.SCOREVISION_API
    if not api_url:
        raise RuntimeError("Neither PRIVATE_GT_API_URL nor SCOREVISION_API is configured")

    params = build_validator_query_params(keypair) if keypair is not None else {}
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
        payload_frames_raw = _coerce_payload_frames(payload)
        payload_frames = [ChallengeFrame(**frame) for frame in payload_frames_raw] or None

        if not challenge_id or (not video_url and not payload_frames):
            logger.warning("Challenge missing task_id or challenge asset (video_url/frames), retrying")
            continue

        try:
            if keypair is not None:
                await complete_task_assignment(
                    challenge_id=challenge_id,
                    keypair=keypair,
                    element_id=element_id,
                )
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
            payload_frames=payload_frames,
            ground_truth=ground_truth,
        )

    logger.warning("No valid challenge found after %d attempts", max_retries)
    return None
