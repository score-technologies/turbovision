from typing import Any
from time import monotonic
from json import loads, dumps
from random import uniform
from logging import getLogger
from typing import AsyncGenerator

from asyncio import TimeoutError, sleep, gather, wait_for
from aiohttp import ClientError

from scorevision.miner.open_source.chute_template.schemas import TVPredictInput, TVPredictOutput
from scorevision.utils.data_models import SVRunOutput, SVPredictResult
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client, get_semaphore
from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.utils.chutes_helpers import validate_chute_integrity
from scorevision.utils.miner_registry import is_registry_bypass

logger = getLogger(__name__)

async def call_miner_model_on_chutes(
    slug: str,
    chute_id: str,
    payload: TVPredictInput,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    miner_uid: int | None = None,
    miner_hotkey: str | None = None,
) -> SVRunOutput:
    logger.info("Verifying chute model is valid")

    if is_registry_bypass(miner_uid, miner_hotkey):
        logger.info(
            "[Integrity] bypass enabled for uid=%s hotkey=%s",
            miner_uid,
            miner_hotkey,
        )
        res = await predict_sv(payload=payload, slug=slug, chute_id=chute_id)
        lat_ms = res.latency_seconds * 1000.0
        return SVRunOutput(
            success=res.success,
            latency_ms=lat_ms,
            predictions=res.predictions if res.success else None,
            error=res.error,
            model=res.model or expected_model,
            latency_p50_ms=lat_ms,
            latency_p95_ms=lat_ms,
            latency_p99_ms=lat_ms,
            latency_max_ms=lat_ms,
        )

    trustworthy, hf_repo_name, hf_repo_revision = await validate_chute_integrity(
        chute_id=chute_id
    )

    mismatch = False
    mismatch_reasons: list[str] = []

    if expected_model:
        if hf_repo_name:
            if hf_repo_name != expected_model:
                mismatch = True
                mismatch_reasons.append(
                    f"model mismatch (on-chain={expected_model}, miner={hf_repo_name})"
                )
        else:
            mismatch = True
            mismatch_reasons.append(
                "missing HF_REPO_NAME in miner code while on-chain model is set"
            )

    if expected_revision:
        if hf_repo_revision:
            if hf_repo_revision != expected_revision:
                mismatch = True
                mismatch_reasons.append(
                    f"revision mismatch (on-chain={expected_revision}, miner={hf_repo_revision})"
                )
        else:
            mismatch = True
            mismatch_reasons.append(
                "missing HF_REPO_REVISION in miner code while on-chain revision is set"
            )

    if not trustworthy or mismatch:
        reason_parts: list[str] = []
        if not trustworthy:
            reason_parts.append("chute integrity hash mismatch")
        if mismatch_reasons:
            reason_parts.extend(mismatch_reasons)
        error_msg = " / ".join(reason_parts) or "Chute integrity check failed"

        logger.error("Chute integrity check failed: %s", error_msg)

        return SVRunOutput(
            success=False,
            latency_ms=0.0,
            predictions=None,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            latency_max_ms=0.0,
            error=error_msg,
            model=hf_repo_name or expected_model,
        )

    res = await predict_sv(payload=payload, slug=slug, chute_id=chute_id)
    lat_ms = res.latency_seconds * 1000.0

    return SVRunOutput(
        success=res.success,
        latency_ms=lat_ms,
        predictions=res.predictions if res.success else None,
        error=res.error,
        model=res.model,
        latency_p50_ms=lat_ms,
        latency_p95_ms=lat_ms,
        latency_p99_ms=lat_ms,
        latency_max_ms=lat_ms,
    )


async def predict_sv(
    payload: TVPredictInput, slug: str, chute_id: str | None = None
) -> SVPredictResult:
    settings = get_settings()

    base_url = settings.CHUTES_MINER_BASE_URL_TEMPLATE.format(
        slug=slug,
    )
    url = f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}"
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
    cold_start_deadline = None
    ordinary_failures = 0
    attempt = 0
    while True:
        attempt += 1
        logger.info(f"Attempt {attempt} to {url}")
        t0_attempt = monotonic()
        cold_start_response = False
        try:
            async def request_once():
                async with semaphore:
                    async with session.post(
                        url, headers=headers, json=payload_json
                    ) as response:
                        return response.status, await response.text()

            if cold_start_deadline is None:
                status, body = await request_once()
            else:
                remaining = cold_start_deadline - monotonic()
                if remaining <= 0:
                    break
                status, body = await wait_for(request_once(), timeout=remaining)

            text = body
            if status == 200:
                data = loads(text)  # TVPredictOutput
                return SVPredictResult(
                    success=bool(data.get("success", True)),
                    model=data.get("model"),
                    latency_seconds=monotonic() - t0_attempt,
                    predictions=data.get("predictions") or data.get("data"),
                    error=data.get("error"),
                    raw=data,
                )
            elif status == 429:
                last_err = f"busy:{text[:120]}"
                logger.error(last_err)
                raise RuntimeError("busy")
            elif 400 <= status < 500:
                last_err = f"{status}:{text[:300]}"
                logger.error(last_err)
                return SVPredictResult(
                    success=False,
                    model=None,
                    latency_seconds=monotonic() - t0_attempt,
                    predictions=None,
                    error=last_err,
                )
            elif status == 503 and "no instances available" in text.lower():
                cold_start_response = True
                last_err = f"chute cold:{text[:120]}"
                logger.warning(last_err)
                if cold_start_deadline is None:
                    cold_start_deadline = monotonic() + max(
                        0, min(30, settings.SCOREVISION_COLD_START_TIMEOUT_S)
                    )
            else:
                last_err = f"HTTP {status}: {text[:300]}"
                logger.error(last_err)
                raise RuntimeError(last_err)

        except TimeoutError as e:
            if cold_start_deadline is None or monotonic() < cold_start_deadline:
                last_err = f"timeout:{e}"
                cold_start_response = cold_start_deadline is not None
            logger.error(last_err)
        except ClientError as e:
            last_err = f"client_error:{type(e).__name__}:{e}"
            logger.error(last_err)
        except Exception as e:
            last_err = f"error:{type(e).__name__}:{e}"
            logger.error(last_err)

        if not cold_start_response:
            ordinary_failures += 1
            if ordinary_failures > retries:
                break

        if cold_start_deadline is not None and monotonic() >= cold_start_deadline:
            break

        sleep_s = backoff * (2 ** min(attempt - 1, 4))
        sleep_s *= 1.0 + uniform(-0.15, 0.15)
        sleep_s = max(0.05, sleep_s)
        if cold_start_deadline is not None:
            sleep_s = min(sleep_s, cold_start_deadline - monotonic())
        if sleep_s > 0:
            logger.info(f"waiting for {sleep_s}s")
            await sleep(sleep_s)

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
