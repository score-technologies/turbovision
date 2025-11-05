from logging import getLogger
from collections import defaultdict

from numpy import logical_and, logical_or, ndarray

from scorevision.vlm_pipeline.utils.response_models import BoundingBox
from scorevision.vlm_pipeline.image_annotation.pairwise import bboxes_to_mask
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


def filter_low_quality_pseudo_gt_annotations(
    annotations: list[PseudoGroundTruth], min_iou_threshold: float = 0.7
) -> list[PseudoGroundTruth]:
    settings = get_settings()
    pgt_lookup = {pgt.frame_number: pgt for pgt in annotations}
    start = min(pgt_lookup.keys())
    end = max(pgt_lookup.keys())
    high_quality_annotations = [pgt_lookup[start]]
    for frame_number in range(start + 1, end + 1):
        pgt_current = pgt_lookup.get(frame_number)
        pgt_prev = pgt_lookup.get(frame_number - 1)
        if pgt_current and pgt_prev:
            transition_similarity = iou_bboxes(
                bboxes1=pgt_current.annotation.bboxes,
                bboxes2=pgt_prev.annotation.bboxes,
                image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                image_width=settings.SCOREVISION_IMAGE_WIDTH,
            )
            logger.info(
                f"Transition similarity IoU for frame {frame_number} = {transition_similarity}"
            )
            if transition_similarity >= min_iou_threshold:
                high_quality_annotations.append(pgt_current)
        else:
            logger.error(f"No Pseudo GT found for prev frame id {frame_number}")

    return high_quality_annotations


def intersection_over_union(mask1: ndarray, mask2: ndarray) -> float:
    intersection_mask = logical_and(mask1, mask2)
    union_mask = logical_or(mask1, mask2)
    intersection = intersection_mask.sum()
    union = union_mask.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def iou_bboxes(
    bboxes1: list[BoundingBox],
    bboxes2: list[BoundingBox],
    image_height: int,
    image_width: int,
) -> float:
    mask1 = bboxes_to_mask(
        bboxes=bboxes1,
        image_height=image_height,
        image_width=image_width,
    )
    mask2 = bboxes_to_mask(
        bboxes=bboxes2,
        image_height=image_height,
        image_width=image_width,
    )
    return intersection_over_union(mask1=mask1, mask2=mask2)


def bbox_jerkiness(ious: list[float]) -> float:
    """Jerkiness is defined as the mean absolute change in IoU between consecutive frames
    (lower is better, indicating stable tracking or motion)."""
    return sum(abs(ious[i + 1] - ious[i]) for i in range(len(ious) - 1)) / (
        len(ious) - 1
    )


def bbox_smoothness(
    video_bboxes: list[list[BoundingBox]], image_height: int, image_width: int
) -> float:
    """
    High Frame Transition IoU => Smoother. Low Frame Transition IoU => Sudden Change

    Smoothness = Mean IoU / (1 + Jerkiness)
    """
    ious = [
        iou_bboxes(
            bboxes1=video_bboxes[t],
            bboxes2=video_bboxes[t + 1],
            image_height=image_height,
            image_width=image_width,
        )
        for t in range(len(video_bboxes) - 1)
    ]

    if len(ious) < 2:
        return 0.0
    logger.info(f"Frame Transition IoUs:{ious}")
    mean_iou = sum(ious) / len(ious)
    jerkiness = bbox_jerkiness(ious=ious)
    smoothness = mean_iou / (1 + jerkiness)
    logger.info(f"Mean IoU: {mean_iou}")
    logger.info(f"Jerkiness: {jerkiness}")
    logger.info(f"Smoothness: {smoothness}")
    return smoothness


def bbox_smoothness_per_type(
    video_bboxes: list[list[BoundingBox]],
    image_height: int,
    image_width: int,
) -> float:
    """
    Computes smoothness for each object type separately using bbox_smoothness(),
    then aggregates them (mean or weighted mean).
    """
    by_type: dict[int, list[list[BoundingBox]]] = defaultdict(list)
    for frame_bboxes in video_bboxes:
        grouped = defaultdict(list)
        for bbox in frame_bboxes:
            grouped[f"{bbox.label.value}_{bbox.cluster_id.value}"].append(bbox)
        for cls_id, boxes in grouped.items():
            by_type[cls_id].append(boxes)

    per_type_scores: dict[int, float] = {}

    for cls_id, frames in by_type.items():
        if len(frames) < 2:
            per_type_scores[cls_id] = 0.0
            continue

        score = bbox_smoothness(
            video_bboxes=frames,
            image_height=image_height,
            image_width=image_width,
        )
        per_type_scores[cls_id] = score
        logger.info(
            f"[bbox_smoothness_per_type] cls_id={cls_id} -> smoothness={score:.4f}"
        )

    if not per_type_scores:
        return 0.0

    logger.info(f"Per-type smoothness: {per_type_scores}")
    overall = sum(per_type_scores.values()) / len(per_type_scores)
    logger.info(f"Overall smoothness (mean): {overall:.4f}")
    return overall
