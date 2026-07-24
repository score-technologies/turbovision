import logging
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 25 * 1024 * 1024


async def download_image(url: str) -> Path:
    logger.info("Downloading image: %s", url)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    content = response.content
    if not content:
        raise ValueError("Downloaded image is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Downloaded image exceeds {MAX_IMAGE_BYTES} bytes")

    content_type = response.headers.get("content-type", "").lower()
    if (
        content_type
        and not content_type.startswith("image/")
        and "application/octet-stream" not in content_type
    ):
        raise ValueError(f"Unexpected image content type: {content_type}")

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        temp_file.write(content)
    finally:
        temp_file.close()
    return Path(temp_file.name)


def delete_image(image_path: Path) -> None:
    try:
        image_path.unlink()
        logger.info("Deleted image: %s", image_path)
    except Exception as exc:
        logger.warning("Failed to delete image %s: %s", image_path, exc)
