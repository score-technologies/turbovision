import logging
from dataclasses import dataclass
from time import perf_counter
import asyncio
import httpx
from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChallengeAttempt:
    response: ChallengeResponse | None
    elapsed_s: float
    timed_out: bool


async def send_challenge(
    miner: RegisteredMiner,
    challenge: Challenge,
    hotkey,
    timeout: float = 30.0,
) -> ChallengeAttempt:
    url = f"http://{miner.ip}:{miner.port}/challenge"
    request = ChallengeRequest(
        challenge_id=challenge.challenge_id,
        video_url=challenge.video_url,
    )
    start = perf_counter()

    try:
        payload_bytes = request.model_dump_json().encode()
        headers = build_signed_headers(hotkey, payload_bytes)
        headers["X-Miner-Hotkey"] = miner.hotkey

        client_timeout = httpx.Timeout(timeout=timeout)
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            response = await asyncio.wait_for(
                client.post(
                    url,
                    json=request.model_dump(),
                    headers=headers,
                ),
                timeout=timeout,
            )
            response.raise_for_status()
            elapsed_s = perf_counter() - start
            if elapsed_s > timeout:
                logger.warning(
                    "Challenge to %s exceeded timeout %.2fs (elapsed %.2fs)",
                    miner.hotkey,
                    timeout,
                    elapsed_s,
                )
                return ChallengeAttempt(
                    response=None,
                    elapsed_s=elapsed_s,
                    timed_out=True,
                )
            return ChallengeAttempt(
                response=ChallengeResponse(**response.json()),
                elapsed_s=elapsed_s,
                timed_out=False,
            )
    except (asyncio.TimeoutError, httpx.TimeoutException):
        elapsed_s = perf_counter() - start
        logger.warning("Challenge to %s timed out after %.2fs", miner.hotkey, elapsed_s)
        return ChallengeAttempt(
            response=None,
            elapsed_s=elapsed_s,
            timed_out=True,
        )
    except Exception as e:
        elapsed_s = perf_counter() - start
        logger.error("Challenge to %s failed: %s", miner.hotkey, e)
        return ChallengeAttempt(
            response=None,
            elapsed_s=elapsed_s,
            timed_out=True,
        )
