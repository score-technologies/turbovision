from logging import getLogger

from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA, circle, fillPoly, polylines, putText, rectangle
import numpy as np
from numpy import ndarray

from scorevision.vlm_pipeline.utils.geometry import (
    AnnotationGeometry,
    AnnotationGeometryType,
    geometry_to_bbox,
)
from scorevision.vlm_pipeline.utils.response_models import (
    BoundingBox,
    FrameAnnotation,
    ShirtColor,
)

logger = getLogger(__name__)


COLOURS = {
    ShirtColor.WHITE: (255, 255, 255),
    ShirtColor.BLACK: (0, 0, 0),
    ShirtColor.RED: (0, 0, 255),
    ShirtColor.BLUE: (255, 0, 0),
    ShirtColor.YELLOW: (0, 255, 255),
    ShirtColor.GREEN: (0, 255, 0),
    ShirtColor.ORANGE: (0, 165, 255),
    ShirtColor.PURPLE: (128, 0, 128),
    ShirtColor.MAROON: (0, 0, 128),
    ShirtColor.PINK: (203, 192, 255),
    ShirtColor.GREY: (128, 128, 128),
    ShirtColor.BROWN: (42, 42, 165),
    ShirtColor.GOLD: (0, 215, 255),
    ShirtColor.SILVER: (192, 192, 192),
    ShirtColor.TURQUOISE: (208, 224, 64),
    ShirtColor.OTHER: (0, 255, 128),
}


def annotate_frame_label(frame: ndarray, label: str) -> None:
    putText(
        frame,
        label,
        (10, 30),
        FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def _geometry_color(bbox: BoundingBox) -> tuple[int, int, int]:
    color = COLOURS[bbox.cluster_id]
    return color


def annotate_bbox(frame: ndarray, bbox: BoundingBox) -> None:
    if bbox.geometry is None:
        return

    color = _geometry_color(bbox)
    geometry = bbox.geometry
    x_min, y_min, x_max, y_max = geometry_to_bbox(geometry)

    if geometry.type == AnnotationGeometryType.BBOX:
        rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    elif geometry.type == AnnotationGeometryType.POINT:
        pt = geometry.points[0]
        circle(frame, (int(round(pt.x)), int(round(pt.y))), 4, color, -1, lineType=LINE_AA)
    else:
        pts = np.array(
            [[int(round(p.x)), int(round(p.y))] for p in geometry.points],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=LINE_AA)
            overlay = frame.copy()
            fillPoly(overlay, [pts], color)
            frame[:] = np.where(overlay > 0, overlay, frame)

    label = bbox.label or ""
    if label:
        putText(
            frame,
            label,
            (x_min, max(0, y_min - 4)),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=LINE_AA,
        )


def annotate_image(image: ndarray, annotations: FrameAnnotation, name: str) -> ndarray:
    annotated_image = image.copy()
    annotate_frame_label(
        frame=annotated_image, label=f"{name}: {annotations.category.value}"
    )
    for bbox in annotations.bboxes:
        annotate_bbox(frame=annotated_image, bbox=bbox)
    return annotated_image
