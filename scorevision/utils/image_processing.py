from PIL import Image
from numpy import ndarray, stack
from io import BytesIO
from base64 import b64encode
from logging import getLogger

from cv2 import imencode
from numpy import ndarray

logger = getLogger(__name__)


def pil_from_array(array: ndarray) -> Image.Image:
    """
    Converts a frame array (H,W,3 or H,W,4) to PIL Image without resizing.
    """
    if array.ndim == 2:
        # grayscale → convert to RGB for consistency
        array = stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        # drop alpha if present
        array = array[..., :3]
    return Image.fromarray(array)


def image_to_base64(img: Image.Image, fmt: str, quality: int, optimise: bool) -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt, quality=quality, optimise=optimise)
    return b64encode(buffer.getvalue()).decode("ascii")


def image_to_b64string(image: ndarray) -> str | None:
    try:
        _, image_buffer = imencode(".png", image)
        b64_image = b64encode(image_buffer.tobytes()).decode("utf-8")
        return b64_image
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")


def images_to_b64strings(images: list[ndarray]) -> list[str]:
    b64_images = []
    for image in images:
        b64_image = image_to_b64string(image=image)
        if b64_image:
            b64_images.append(b64_image)
    return b64_images
