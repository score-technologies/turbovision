from io import BytesIO
from base64 import b64decode
from typing import Any

from PIL import Image
from pydantic import BaseModel


# ======NOTE: These must match what is in the chute ==========
class TVPredictInput(BaseModel):
    url: str
    meta: dict[str, Any] = {}


class TVPredictOutput(BaseModel):
    success: bool
    predictions: dict[str, list[dict]] | None = None
    error: str | None = None


# ==============================================================


class SVFrame(BaseModel):
    frame_id: int
    data: str  # base64 encoded image

    @property
    def image(self) -> Image.Image:
        return Image.open(BytesIO(b64decode(self.data))).convert("RGB")


class SVBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float = 1.0


class SVFrameResult(BaseModel):
    frame_id: int
    boxes: list[SVBox]
    keypoints: list[tuple[int, int]]  # pixel coordinates
    # action:str #TODO:
