from typing import Any
from time import monotonic
from json import loads, dumps
from random import uniform
from logging import getLogger
from typing import AsyncGenerator
from urllib.parse import quote

from asyncio import TimeoutError, sleep, gather
from aiohttp import ClientError

from scorevision.chute_template.schemas import TVPredictInput, TVPredictOutput
from scorevision.utils.data_models import SVRunOutput, SVPredictResult
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client, get_semaphore
from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.utils.chutes_helpers import (
    get_chute_slug_and_id,
    warmup_chute,
    validate_chute_integrity,
)

logger = getLogger(__name__)


async def call_miner_model_on_chutes(
    slug: str, chute_id: str, payload: TVPredictInput
) -> SVRunOutput:
    logger.info("Verifying chute model is valid")
    trustworthy = await validate_chute_integrity(chute_id=chute_id)
    if not trustworthy:
        logger.error("Chute integrity check failed")
        return SVRunOutput(
            success=False,
            latency_ms=0.0,
            predictions=None,
            error="Chute integrity check failed",
            model=None,
        )
    res = await predict_sv(payload=payload, slug=slug, chute_id=chute_id)
    return SVRunOutput(
        success=res.success,
        latency_ms=res.latency_seconds * 1000.0,
        predictions=res.predictions if res.success else None,
        error=res.error,
        model=res.model,
    )


async def predict_sv(
    payload: TVPredictInput, slug: str, chute_id: str | None = None
) -> SVPredictResult:
    settings = get_settings()

    base_url = settings.CHUTES_MINER_BASE_URL_TEMPLATE.format(
        slug=slug,
    )
    url_ = f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}"
    url = f"{settings.CHUTES_MINERS_ENDPOINT}/misc/proxy?url={quote(url_, safe='')}"
    api_key = settings.CHUTES_API_KEY
    retries = settings.SCOREVISION_API_N_RETRIES
    backoff = settings.SCOREVISION_BACKOFF_RATE

    if not api_key.get_secret_value():
        return SVPredictResult(
            success=False,
            model=None,
            latency_seconds=0.0,
            predictions=None,
            error="CHUTES_API_KEY missing",
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.get_secret_value()}",
    }

    session = await get_async_client()
    semaphore = get_semaphore()
    t0 = monotonic()
    last_err = None

    payload_json = payload.model_dump(mode="json")
    for attempt in range(1, retries + 2):
        logger.info(f"Attempt {attempt} to {url}")
        try:
            async with semaphore:
                async with session.post(
                    url, headers=headers, json=payload_json
                ) as response:
                    logger.info(f"request status: {response.status}")
                    text = await response.text()
                    if response.status == 200:
                        data = loads(text)  # TVPredictOutput
                        return SVPredictResult(
                            success=bool(data.get("success", True)),
                            model=data.get("model"),
                            latency_seconds=monotonic() - t0,
                            predictions=data.get("predictions") or data.get("data"),
                            error=data.get("error"),
                            raw=data,
                        )
                    elif response.status == 429:
                        last_err = f"busy:{text[:120]}"
                        logger.error(last_err)
                        raise RuntimeError("busy")
                    elif 400 <= response.status < 500:
                        last_err = f"{response.status}:{text[:300]}"
                        logger.error(last_err)
                        return SVPredictResult(
                            success=False,
                            model=None,
                            latency_seconds=monotonic() - t0,
                            predictions=None,
                            error=last_err,
                        )
                    elif response.status == 503:
                        last_err = f"chute cold:{text[:120]}"
                        logger.error(last_err)
                        if chute_id:
                            await warmup_chute(chute_id=chute_id)
                        raise RuntimeError(last_err)
                    else:
                        last_err = f"HTTP {response.status}: {text[:300]}"
                        logger.error(last_err)
                        raise RuntimeError(last_err)

        except TimeoutError as e:
            last_err = f"timeout:{e}"
            logger.error(last_err)
        except ClientError as e:
            last_err = f"client_error:{type(e).__name__}:{e}"
            logger.error(last_err)
        except Exception as e:
            last_err = f"error:{type(e).__name__}:{e}"
            logger.error(last_err)

        if attempt <= retries:
            sleep_s = backoff * (2 ** (attempt - 1))
            sleep_s *= 1.0 + uniform(-0.15, 0.15)
            logger.info(f"waiting for {sleep_s}s")
            await sleep(max(0.05, sleep_s))

    return SVPredictResult(
        success=False,
        model=None,
        latency_seconds=monotonic() - t0,
        predictions=None,
        error=last_err or "unknown_error",
    )


async def _warmup_from_video(
    *,
    video_url: str,
    slug: str = "demo",
    base_url: str | None = None,
):
    settings = get_settings()

    fake_chal = {
        "task_id": "warmup-fixed",
        "video_url": video_url,
        "fps": settings.SCOREVISION_VIDEO_FRAMES_PER_SECOND,
        "seed": 0,
    }

    payload, _, _, _, frame_store = await prepare_challenge_payload(challenge=fake_chal)

    async def _one():
        try:
            await predict_sv(
                payload=payload,
                slug=slug,
            )
        except Exception as e:
            logger.debug(f"warmup call error: {e}")

    await gather(*(_one() for _ in range(max(1, settings.SCOREVISION_WARMUP_CALLS))))
    frame_store.unlink()


async def warmup(url: str, slug: str) -> None:
    try:
        await _warmup_from_video(
            video_url=url,
            slug=slug,
        )
        logger.info("Warmup done.")
    except Exception as e:
        logger.error(f"Warmup errored (non-fatal): {type(e).__name__}: {e}")
