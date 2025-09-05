from numpy import logical_and, logical_or, ndarray, uint8, where, zeros

from scorevision.vlm_pipeline.utils.response_models import BoundingBox


def bboxes_to_mask(
    bboxes: list[BoundingBox], image_height: int, image_width: int
) -> ndarray:
    mask = zeros((image_height, image_width), dtype=uint8)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.bbox_2d
        mask[y_min:y_max, x_min:x_max] = 1
    return mask


def display_iou(
    bboxes_a: list[BoundingBox],
    bboxes_b: list[BoundingBox],
    image_height: int,
    image_width: int,
) -> ndarray:
    mask_a = bboxes_to_mask(
        bboxes=bboxes_a, image_height=image_height, image_width=image_width
    )
    mask_b = bboxes_to_mask(
        bboxes=bboxes_b, image_height=image_height, image_width=image_width
    )

    mask_intersection = logical_and(mask_a, mask_b)
    mask_union = logical_or(mask_a, mask_b)

    mask_iou = zeros((*mask_a.shape, 3), dtype=uint8)
    mask_iou[..., 0] = where((mask_a == 1) & (mask_b == 0), 255, 0)
    mask_iou[..., 1] = where((mask_a == 0) & (mask_b == 1), 255, 0)
    mask_iou[..., 0] = where(mask_intersection, 255, mask_iou[..., 0])
    mask_iou[..., 1] = where(mask_intersection, 255, mask_iou[..., 1])

    return mask_iou
