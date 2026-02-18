import logging
import tempfile
from pathlib import Path
import cv2
import httpx

logger = logging.getLogger(__name__)


async def download_video(url: str) -> Path:
    logger.info(f"Downloading video: {url}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(response.content)
        temp_file.close()

        return Path(temp_file.name)


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def delete_video(video_path: Path) -> None:
    try:
        video_path.unlink()
        logger.info(f"Deleted video: {video_path}")
    except Exception as e:
        logger.warning(f"Failed to delete video: {video_path}, error: {e}")

