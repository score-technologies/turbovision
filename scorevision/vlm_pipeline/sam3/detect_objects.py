import asyncio
from logging import getLogger

from numpy import ndarray

from scorevision.utils.image_processing import image_to_b64string
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.chutes_helpers import warmup_chute
from scorevision.vlm_pipeline.sam3.schemas import Sam3Result

logger = getLogger(__name__)


async def sam3_chute(
    image: ndarray, object_names: list[str], threshold: float, mosaic: int = 0
) -> list[Sam3Result]:
    settings = get_settings()
    endpoint = settings.CHUTES_SAM3_ENDPOINT
    logger.warning(f"SAM3 endpoint={endpoint}")
    logger.warning(f"prompts={object_names} thresh={threshold} mosaic={mosaic}")
    headers = {
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
        "Content-Type": "application/json",
    }
    payload = {
        "image": {"type": "base64", "value": image_to_b64string(image=image)},
        "prompts": [{"type": "text", "text": o} for o in object_names],
        "output_prob_thresh": threshold,
        "mosaic": mosaic,
    }
    b64 = image_to_b64string(image=image)
    logger.warning(f"SAM3 image b64 len={len(b64)} chars")
    payload["image"]["value"] = b64
    last_exc: Exception | None = None

    for attempt in range(settings.SCOREVISION_API_N_RETRIES):
        logger.info(
            "Calling SAM3 via Chutes attempt %s/%s...",
            attempt + 1,
            settings.SCOREVISION_API_N_RETRIES,
        )
        session = await get_async_client()

        try:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                text = await response.text()

                if response.status == 200:
                    response_json = await response.json()
                    message_content = response_json.get("prompt_results", [])
                    return [Sam3Result(**r) for r in message_content]

                if 500 <= response.status < 600:
                    try:
                        await warmup_chute(chute_id=settings.CHUTES_SAM3_ID)
                    except Exception as e:
                        logger.warning("warmup_chute failed: %s", e)

                raise Exception(f"API request failed with status {response.status}: {text}")

        except Exception as e:
            last_exc = e
            base = max(1.0, float(settings.SCOREVISION_API_RETRY_DELAY_S))
            wait_time = min(base, 2 ** attempt)
            logger.info("API request failed: %s. Retrying in %.1f s...", e, wait_time)
            await asyncio.sleep(wait_time)

    raise last_exc or RuntimeError("SAM3 request failed (unknown error)")