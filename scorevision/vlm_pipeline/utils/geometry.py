from __future__ import annotations

from enum import StrEnum
from math import floor, ceil

import numpy as np
from cv2 import fillPoly
from numpy import ndarray, zeros
from pydantic import BaseModel, Field, model_validator


class AnnotationGeometryType(StrEnum):
    BBOX = "bbox"
    POLYGON = "polygon"
    POINT = "point"


class Point2D(BaseModel):
    x: float
    y: float


class AnnotationGeometry(BaseModel):
    type: AnnotationGeometryType
    points: list[Point2D] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_points(self):
        n = len(self.points)
        if self.type == AnnotationGeometryType.BBOX and n != 2:
            raise ValueError("bbox geometry requires exactly 2 points")
        if self.type == AnnotationGeometryType.POLYGON and n < 3:
            raise ValueError("polygon geometry requires at least 3 points")
        if self.type == AnnotationGeometryType.POINT and n != 1:
            raise ValueError("point geometry requires exactly 1 point")
        return self


def bbox_points_to_geometry(
    bbox_2d: tuple[int, int, int, int] | list[int] | list[float],
) -> AnnotationGeometry:
    x1, y1, x2, y2 = bbox_2d
    return AnnotationGeometry(
        type=AnnotationGeometryType.BBOX,
        points=[
            Point2D(x=float(x1), y=float(y1)),
            Point2D(x=float(x2), y=float(y2)),
        ],
    )


def geometry_to_bbox(geometry: AnnotationGeometry) -> tuple[int, int, int, int]:
    if geometry.type == AnnotationGeometryType.BBOX:
        p1, p2 = geometry.points
        return (
            int(floor(min(p1.x, p2.x))),
            int(floor(min(p1.y, p2.y))),
            int(ceil(max(p1.x, p2.x))),
            int(ceil(max(p1.y, p2.y))),
        )

    xs = [p.x for p in geometry.points]
    ys = [p.y for p in geometry.points]
    return (
        int(floor(min(xs))),
        int(floor(min(ys))),
        int(ceil(max(xs))),
        int(ceil(max(ys))),
    )


def geometry_to_mask(
    geometry: AnnotationGeometry, image_height: int, image_width: int
) -> ndarray:
    mask = zeros((image_height, image_width), dtype=np.uint8)
    if geometry.type == AnnotationGeometryType.POINT:
        point = geometry.points[0]
        x = int(round(point.x))
        y = int(round(point.y))
        if 0 <= x < image_width and 0 <= y < image_height:
            mask[y, x] = 1
        return mask

    if geometry.type == AnnotationGeometryType.BBOX:
        x_min, y_min, x_max, y_max = geometry_to_bbox(geometry)
        mask[y_min:y_max, x_min:x_max] = 1
        return mask

    points = np.array([[int(round(p.x)), int(round(p.y))] for p in geometry.points], dtype=np.int32)
    if len(points) >= 3:
        fillPoly(mask, [points], 1)
    return mask
