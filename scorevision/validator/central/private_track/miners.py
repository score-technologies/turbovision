import logging
import httpx
from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner

logger = logging.getLogger(__name__)


async def send_challenge(
    miner: RegisteredMiner,
    challenge: Challenge,
    hotkey,
    timeout: float = 120.0,
) -> ChallengeResponse | None:
    url = f"http://{miner.ip}:{miner.port}/challenge"
    request = ChallengeRequest(
        challenge_id=challenge.challenge_id,
        video_url=challenge.video_url,
    )

    try:
        payload_bytes = request.model_dump_json().encode()
        headers = build_signed_headers(hotkey, payload_bytes)
        headers["X-Miner-Hotkey"] = miner.hotkey

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=request.model_dump(),
                headers=headers,
            )
            response.raise_for_status()
            return ChallengeResponse(**response.json())
    except httpx.TimeoutException:
        logger.warning("Challenge to %s timed out", miner.hotkey)
        return None
    except Exception as e:
        logger.error("Challenge to %s failed: %s", miner.hotkey, e)
        return None
