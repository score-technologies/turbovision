import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import aiohttp

logger = logging.getLogger(__name__)


async def fetch_json_from_url(url: str, timeout_s: int = 30) -> Any | None:
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("HTTP %d from %s", resp.status, url)
                    return None
                return await resp.json()
    except Exception as e:
        logger.error("Error fetching %s: %s", url, e)
        return None


async def fetch_head_metadata(
    url: str, timeout_s: int = 10
) -> tuple[str | None, str | None]:
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url) as resp:
                if resp.status >= 400:
                    return None, None
                return (resp.headers.get("ETag"), resp.headers.get("Last-Modified"))
    except Exception:
        return None, None


def build_index_url(public_url: str) -> str:
    url = public_url.rstrip("/")
    if url.endswith("/index.json"):
        return url
    if url.endswith("/scorevision"):
        return f"{url}/index.json"
    if "/scorevision" in url:
        base = url.split("/scorevision")[0]
        return f"{base}/scorevision/index.json"
    return f"{url}/scorevision/index.json"


def extract_base_url(public_url: str) -> str:
    url = public_url.rstrip("/")
    if "/scorevision" in url:
        return url.split("/scorevision")[0]
    return url


def bucket_base_from_index(index_url: str) -> str:
    u = urlparse(index_url)
    return f"{u.scheme}://{u.netloc}/"


def normalize_index_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.strip().endswith(".json"):
        return url.strip()
    return url.rstrip("/") + "/scorevision/index.json"


def build_public_index_url_from_base(public_base: str | None) -> str | None:
    if not public_base:
        return None
    return public_base.rstrip("/") + "/scorevision/index.json"


async def fetch_index_keys(public_url: str) -> list[str]:
    index_url = build_index_url(public_url)
    logger.info("Fetching index from: %s", index_url)

    data = await fetch_json_from_url(index_url)
    if data is None:
        logger.error("Failed to fetch index (got None)")
        return []

    if isinstance(data, list):
        logger.info("Index contains %d entries", len(data))
        return [k for k in data if isinstance(k, str)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        entries = data.get("entries", [])
        logger.info("Index contains %d entries", len(entries))
        return [e.get("path") for e in entries if isinstance(e.get("path"), str)]

    logger.error("Unexpected index format: %s", type(data))
    return []


async def fetch_shard_lines(public_url: str, key: str) -> list[dict]:
    if key.startswith("http"):
        url = key
    else:
        base = extract_base_url(public_url)
        url = f"{base}/{key}"

    data = await fetch_json_from_url(url)
    if data is None:
        return []

    if isinstance(data, list):
        return data
    return [data] if isinstance(data, dict) else []


async def fetch_responses_data(
    responses_key: str, public_url: str
) -> tuple[dict[str, Any] | None, str | None]:
    if not responses_key:
        return None, None

    base = extract_base_url(public_url)
    url = f"{base}/{responses_key}"
    logger.info("Fetching miner responses from: %s", url)

    data = await fetch_json_from_url(url)
    if data is None:
        return None, None

    predictions = data.get("predictions")
    video_url = data.get("video_url")
    return predictions, video_url


async def fetch_miner_predictions(
    responses_key: str, public_url: str
) -> dict[str, Any] | None:
    predictions, _ = await fetch_responses_data(responses_key, public_url)
    return predictions


def extract_block_from_key(key: str) -> int | None:
    name = Path(key).name
    try:
        return int(name.split("-", 1)[0])
    except Exception:
        return None


def filter_keys_by_tail(
    keys: list[str], tail_blocks: int
) -> tuple[list[str], int, int]:
    pairs: list[tuple[int, str]] = []
    for key in keys:
        block = extract_block_from_key(key)
        if block is not None:
            pairs.append((block, key))

    if not pairs:
        return [], 0, 0

    pairs.sort()
    max_block = pairs[-1][0]
    min_keep = max_block - tail_blocks
    filtered = [key for (block, key) in pairs if block >= min_keep]

    return filtered, max_block, min_keep

