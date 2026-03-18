import asyncio
import sys
import time
import httpx
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse

READINESS_TIMEOUT_S = 300
READINESS_POLL_S = 3


async def wait_for_ready(miner_url: str) -> bool:
    deadline = time.monotonic() + READINESS_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{miner_url}/docs")
                if response.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(READINESS_POLL_S)
    return False


async def send_challenge(
    challenge_id: str,
    video_url: str,
    timeout_s: float,
    miner_url: str,
    miner_hotkey: str = "",
) -> tuple[ChallengeResponse | None, float, bool]:
    request = ChallengeRequest(challenge_id=challenge_id, video_url=video_url)
    start = time.perf_counter()

    headers = {}
    if miner_hotkey:
        headers["X-Miner-Hotkey"] = miner_hotkey

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=timeout_s)) as client:
            response = await asyncio.wait_for(
                client.post(
                    f"{miner_url}/challenge",
                    json=request.model_dump(),
                    headers=headers,
                ),
                timeout=timeout_s,
            )
            elapsed = time.perf_counter() - start
            response.raise_for_status()
            return ChallengeResponse(**response.json()), elapsed, False

    except (asyncio.TimeoutError, httpx.TimeoutException):
        return None, time.perf_counter() - start, True

    except Exception as e:
        print(f"Challenge error: {e}", file=sys.stderr, flush=True)
        return None, time.perf_counter() - start, False
