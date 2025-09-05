from tempfile import NamedTemporaryFile
from logging import getLogger
from pathlib import Path

from contextlib import contextmanager
from cv2 import (
    CAP_PROP_FRAME_COUNT,
    COLOR_BGR2GRAY,
    COLOR_HSV2BGR,
    NORM_MINMAX,
    VideoCapture,
    calcOpticalFlowFarneback,
    cartToPolar,
    cvtColor,
    normalize,
)
from numpy import ndarray, pi, zeros_like


from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client

logger = getLogger(__name__)


@contextmanager
def open_video(path: Path) -> VideoCapture:
    logger.info(f"Attempting to open video: {path}")
    if not path.exists():
        raise FileNotFoundError
    if not path.is_file():
        raise ValueError("Path is not a file")
    video = VideoCapture(str(path))
    if not video.isOpened():
        video.release()
        raise ValueError("Could not open video")
    try:
        yield video
    finally:
        video.release()


def background_temporal_differencing(
    video_path: Path, frame_numbers: list[int]
) -> tuple[dict[int, ndarray], dict[int, ndarray]]:
    logger.info(
        f"Computing Background Temporal Differencing for frame_numbers {frame_numbers} using Dense Optical Flow..."
    )
    images, flow_images = {}, {}
    with open_video(path=video_path) as video:
        if not video.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        max_frame_number = int(video.get(CAP_PROP_FRAME_COUNT))
        prev_frame, prev_gray = None, None
        for frame_number in range(max_frame_number):
            ok, frame = video.read()
            if not ok:
                logger.error(f"Error reading frame {frame_number}")
                continue
            images[frame_number] = frame

            gray = cvtColor(frame, COLOR_BGR2GRAY)
            if frame_number in frame_numbers and prev_gray is not None:
                flow = calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, ang = cartToPolar(flow[..., 0], flow[..., 1])
                hsv = zeros_like(prev_frame)
                hsv[..., 0] = ang * 180 / pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = normalize(mag, None, 0, 255, NORM_MINMAX)
                rgb = cvtColor(hsv, COLOR_HSV2BGR)

                flow_images[frame_number] = rgb

            prev_gray = gray
            prev_frame = frame

    return images, flow_images


async def download_video(
    url: str, frame_numbers: list[int]
) -> tuple[str, dict[int, ndarray], dict[int, ndarray]]:
    settings = get_settings()
    session = await get_async_client()
    async with session.get(url) as response:
        if response.status != 200:
            txt = await response.text()
            raise RuntimeError(f"Download failed {response.status}: {txt[:200]}")
        data = await response.read()

    with NamedTemporaryFile(prefix="sv_video_", suffix=".mp4") as f:
        f.write(data)

        frames, flows = background_temporal_differencing(
            video_path=Path(f.name), frame_numbers=frame_numbers
        )
    name = url.split("/")[-1]
    return name, frames, flows
