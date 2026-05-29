from io import BytesIO
from base64 import b64decode
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field, model_validator


# ======NOTE: These must match what is in the chute ==========
class TVFrame(BaseModel):
    frame_id: int
    url: str | None = None
    data: str | None = None


class TVPredictInput(BaseModel):
    url: str | None = None
    frames: list[TVFrame] | None = None
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


class GeometryPoint(BaseModel):
    x: float
    y: float


class Geometry(BaseModel):
    type: str
    points: list[GeometryPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_points(self):
        if self.type == "bbox" and len(self.points) != 2:
            raise ValueError("bbox geometry requires exactly 2 points")
        if self.type == "polygon" and len(self.points) < 3:
            raise ValueError("polygon geometry requires at least 3 points")
        if self.type == "point" and len(self.points) != 1:
            raise ValueError("point geometry requires exactly 1 point")
        return self


class SVAnnotation(BaseModel):
    label: str | None = None
    score: float = 1.0
    cluster_id: int | None = None
    geometry: Geometry


class SVFrameResult(BaseModel):
    frame_id: int
    annotations: list[SVAnnotation]
    keypoints: list[tuple[int, int]]  # pixel coordinates
    # action:str #TODO:
