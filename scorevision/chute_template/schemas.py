import os
from io import BytesIO
from typing import Any, Generator
from base64 import b64decode
from traceback import format_exc
from random import randint
from requests import get
from tempfile import NamedTemporaryFile
from pathlib import Path

from pydantic import BaseModel
from PIL import Image
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT
from numpy import ndarray

from huggingface_hub import snapshot_download
from ultralytics import YOLO

from chutes.chute import Chute, NodeSelector
from chutes.image import Image as ChutesImage


class SVFrame(BaseModel):
    frame_id: int
    data: str  # base64 encoded image

    @property
    def image(self) -> Image.Image:
        return Image.open(BytesIO(b64decode(self.data))).convert("RGB")


class SVPredictInput(BaseModel):
    url: str
    meta: dict[str, Any] | None = None


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


class SVPredictOutput(BaseModel):
    success: bool
    model: str
    predictions: dict[str, list[SVFrameResult]] | None = None
    error: str | None = None
