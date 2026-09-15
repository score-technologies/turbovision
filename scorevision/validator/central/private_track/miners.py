import asyncio
import json
import logging
from dataclasses import dataclass
from time import perf_counter

import httpx

from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner

logger = logging.getLogger(__name__)

LARGE_RESPONSE_BYTES = 1_000_000
SLOW_RESPONSE_PARSE_MS = 100.0


class ResponseTooLargeError(Exception):
    def __init__(self, received_bytes: int) -> None:
        self.received_bytes = received_bytes
        super().__init__(
            f"response exceeded {LARGE_RESPONSE_BYTES} bytes "
            f"(received at least {received_bytes})"
        )


async def _post_with_size_limit(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict[str, str],
) -> tuple[httpx.Response, bytearray]:
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > LARGE_RESPONSE_BYTES:
                raise ResponseTooLargeError(declared_bytes)

        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
            next_size = len(body) + len(chunk)
            if next_size > LARGE_RESPONSE_BYTES:
                raise ResponseTooLargeError(next_size)
            body.extend(chunk)

        buffered_response = httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=body,
            request=response.request,
        )
        return buffered_response, body


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
        image_url=challenge.image_url,
        frames=challenge.payload_frames,
    )
    excluded_fields = {"image_url"} if request.image_url is None else set()
    start = perf_counter()

    try:
        payload_bytes = request.model_dump_json(exclude=excluded_fields).encode()
        headers = build_signed_headers(
            hotkey,
            payload_bytes,
            miner_hotkey=miner.hotkey,
        )

        client_timeout = httpx.Timeout(timeout=timeout)
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            response, response_body = await asyncio.wait_for(
                _post_with_size_limit(
                    client,
                    url,
                    request.model_dump(exclude=excluded_fields),
                    headers,
                ),
                timeout=timeout,
            )
            response.raise_for_status()
            http_elapsed_s = perf_counter() - start
            if http_elapsed_s > timeout:
                logger.warning(
                    "Challenge to %s exceeded timeout %.2fs (elapsed %.2fs)",
                    miner.hotkey,
                    timeout,
                    http_elapsed_s,
                )
                return ChallengeAttempt(
                    response=None,
                    elapsed_s=http_elapsed_s,
                    timed_out=True,
                )

            body_bytes = len(response_body)
            content_length = response.headers.get("content-length")
            json_started = perf_counter()
            try:
                response_payload = json.loads(response_body)
            except Exception as exc:
                json_parse_ms = (perf_counter() - json_started) * 1000.0
                logger.warning(
                    "Challenge response diagnostics hotkey=%s status=%s body_bytes=%d "
                    "content_length=%s http_s=%.3f json_parse_ms=%.1f "
                    "validation_ms=not_started total_s=%.3f outcome=json_error error_type=%s",
                    miner.hotkey,
                    response.status_code,
                    body_bytes,
                    content_length,
                    http_elapsed_s,
                    json_parse_ms,
                    perf_counter() - start,
                    type(exc).__name__,
                )
                raise

            json_parse_ms = (perf_counter() - json_started) * 1000.0
            validation_started = perf_counter()
            try:
                parsed_response = ChallengeResponse(**response_payload)
            except Exception as exc:
                validation_ms = (perf_counter() - validation_started) * 1000.0
                logger.warning(
                    "Challenge response diagnostics hotkey=%s status=%s body_bytes=%d "
                    "content_length=%s http_s=%.3f json_parse_ms=%.1f validation_ms=%.1f "
                    "total_s=%.3f outcome=validation_error error_type=%s",
                    miner.hotkey,
                    response.status_code,
                    body_bytes,
                    content_length,
                    http_elapsed_s,
                    json_parse_ms,
                    validation_ms,
                    perf_counter() - start,
                    type(exc).__name__,
                )
                raise

            validation_ms = (perf_counter() - validation_started) * 1000.0
            diagnostic_log = (
                logger.warning
                if body_bytes >= LARGE_RESPONSE_BYTES
                or json_parse_ms >= SLOW_RESPONSE_PARSE_MS
                or validation_ms >= SLOW_RESPONSE_PARSE_MS
                else logger.info
            )
            diagnostic_log(
                "Challenge response diagnostics hotkey=%s status=%s body_bytes=%d "
                "content_length=%s http_s=%.3f json_parse_ms=%.1f validation_ms=%.1f "
                "total_s=%.3f outcome=ok prediction_count=%d",
                miner.hotkey,
                response.status_code,
                body_bytes,
                content_length,
                http_elapsed_s,
                json_parse_ms,
                validation_ms,
                perf_counter() - start,
                parsed_response.prediction_count,
            )
            return ChallengeAttempt(
                response=parsed_response,
                elapsed_s=http_elapsed_s,
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
    except ResponseTooLargeError as e:
        elapsed_s = perf_counter() - start
        logger.warning(
            "Challenge response rejected hotkey=%s outcome=response_too_large "
            "received_bytes=%d max_bytes=%d elapsed_s=%.3f",
            miner.hotkey,
            e.received_bytes,
            LARGE_RESPONSE_BYTES,
            elapsed_s,
        )
        return ChallengeAttempt(
            response=None,
            elapsed_s=elapsed_s,
            timed_out=True,
        )
    except httpx.HTTPStatusError as e:
        elapsed_s = perf_counter() - start
        body = ""
        try:
            body = e.response.text
        except Exception:
            body = "<unavailable>"
        logger.error(
            "Challenge to %s failed with HTTP %s: %s",
            miner.hotkey,
            e.response.status_code,
            body,
        )
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
