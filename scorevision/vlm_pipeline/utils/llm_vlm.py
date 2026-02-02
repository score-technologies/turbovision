from asyncio import sleep as asleep
from contextlib import asynccontextmanager
from functools import wraps
from json import loads
from logging import getLogger
from os import environ
from enum import Enum

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from numpy import ndarray

from scorevision.utils.image_processing import images_to_b64strings
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client

logger = getLogger(__name__)


class VLMProvider(Enum):
    PRIMARY = "Chutes"
    BACKUP = "OpenRouter"


def construct_vlm_input(
    system_prompt: str, user_prompt: str, images: list[ndarray]
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
            + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                }
                for b64_image in images_to_b64strings(images=images)
            ],
        },
    ]


async def async_vlm_api(
    images: list[ndarray],
    system_prompt: str,
    user_prompt: str,
    provider: VLMProvider,
    *,
    model_override: str | None = None,
    temperature_override: float | None = None,
) -> dict:
    """
    Appel VLM générique. Possibilité d'override du modèle et de la température,
    tout en conservant le provider (Chutes/OpenRouter) configuré.
    """
    settings = get_settings()
    if provider == VLMProvider.PRIMARY:
        api_key = settings.CHUTES_API_KEY
        model = model_override if model_override else settings.CHUTES_VLM
        endpoint = settings.CHUTES_VLM_ENDPOINT
    elif provider == VLMProvider.BACKUP:
        api_key = settings.OPENROUTER_API_KEY
        model = model_override if model_override else settings.OPENROUTER_VLM
        endpoint = settings.OPENROUTER_VLM_ENDPOINT
    else:
        raise ValueError(f"Unsupported API provider: {provider.value}")

    headers = {
        "Authorization": f"Bearer {api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    messages = construct_vlm_input(
        system_prompt=system_prompt, user_prompt=user_prompt, images=images
    )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": (
            settings.SCOREVISION_VLM_TEMPERATURE
            if temperature_override is None
            else temperature_override
        ),
        "response_format": {"type": "json_object"},
    }
    logger.error(
        "[VLM PAYLOAD CHECK] model=%s | system_type=%s | user_type=%s | system_preview=%r",
        model,
        type(messages[0]["content"]),
        type(messages[1]["content"]),
        messages[0]["content"][:200] if isinstance(messages[0]["content"], str) else messages[0]["content"],
    )
    session = await get_async_client()
    async with session.post(
        endpoint,
        json=payload,
        headers=headers,
    ) as response:
        if response.status == 200:
            response_json = await response.json()
            logger.info(response_json)
            choices = response_json.get("choices") or []
            if not choices:
                raise ValueError("no choices returned")
            message = choices[0].get("message") or {}
            message_content = message.get("content") or "{}"
            logger.info(message_content)
            return loads(message_content)
        raise Exception(
            f"API request failed with status {response.status}: {await response.text()}"
        )


def retry_api(func):
    """Decorator to retry an async function if it returns None."""
    settings = get_settings()
    providers = []
    if settings.CHUTES_API_KEY.get_secret_value():
        providers.append(VLMProvider.PRIMARY)
    else:
        logger.error(f"No API key set for {VLMProvider.PRIMARY.value}")
    if settings.OPENROUTER_API_KEY.get_secret_value():
        providers.append(VLMProvider.BACKUP)
    else:
        logger.error(f"No API key set for {VLMProvider.BACKUP.value}")

    @wraps(func)
    async def wrapper(*args, **kwargs):
        for provider in providers:
            kwargs["provider"] = provider
            for attempt in range(1, settings.SCOREVISION_API_N_RETRIES + 1):
                logger.info(
                    f"Calling API: {provider.value} attempt {attempt}/{settings.SCOREVISION_API_N_RETRIES}..."
                )
                result = await func(*args, **kwargs)
                if result is not None:
                    return result
                wait_time = min(attempt, settings.SCOREVISION_API_RETRY_DELAY_S)
                logger.info(f"Failed. Retrying in {wait_time} s...")
                await asleep(wait_time)
        return None

    return wrapper
