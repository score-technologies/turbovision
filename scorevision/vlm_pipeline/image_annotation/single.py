from logging import getLogger

from cv2 import FONT_HERSHEY_SIMPLEX, putText, rectangle
from numpy import ndarray

from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation

logger = getLogger(__name__)


COLOURS = [
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 0, 255),  # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 0),  # Dark Green
    (128, 0, 0),  # Dark Blue
]


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


def annotate_bbox(frame: ndarray, bbox: BoundingBox) -> None:
    x_min, y_min, x_max, y_max = bbox.bbox_2d
    color = COLOURS[(bbox.cluster_id - 1) % len(COLOURS)]
    rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    putText(
        frame,
        bbox.label.value,
        (x_min, y_min - 4),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def annotate_image(image: ndarray, annotations: FrameAnnotation, name: str) -> ndarray:
    annotated_image = image.copy()
    annotate_frame_label(
        frame=annotated_image, label=f"{name}: {annotations.category.value}"
    )
    for bbox in annotations.bboxes:
        annotate_bbox(frame=annotated_image, bbox=bbox)
    return annotated_image
