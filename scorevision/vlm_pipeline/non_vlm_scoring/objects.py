from logging import getLogger
from collections import Counter

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import iou_bboxes
from scorevision.utils.settings import get_settings
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
)
from scorevision.vlm_pipeline.utils.response_models import (
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)

logger = getLogger(__name__)


def compare_object_counts(
    pseudo_gt: list[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> float:
    frame_scores = []
    for pgt in pseudo_gt:
        annotations_miner = miner_predictions.get(pgt.frame_number)
        if annotations_miner is None:
            logger.info(
                f"Frame {frame_number} missing annotations_miner ({annotations_miner})"
            )
            frame_score = 0.0
        else:
            object_type_scores = []
            for object_type in ObjectOfInterest:
                pgt_bboxes_for_object_type = [
                    bbox for bbox in pgt.annotation.bboxes if bbox.label == object_type
                ]
                miner_bboxes_for_object_type = [
                    bbox
                    for bbox in annotations_miner["bboxes"]
                    if bbox.label == object_type
                ]
                count_pgt = len(pgt_bboxes_for_object_type)
                count_miner = len(miner_bboxes_for_object_type)
                delta = abs(count_pgt - count_miner)
                object_type_score = 1 - delta / max(count_pgt, count_miner, 1)
                object_type_scores.append(object_type_score)
            frame_score = sum(object_type_scores) / len(object_type_scores)
        logger.info(f"[compare_object_counts] Frame {pgt.frame_number}: {frame_score}")
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)


def compare_team_labels(
    pseudo_gt: list[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> float:
    settings = get_settings()
    frame_scores = []
    for pgt in pseudo_gt:
        annotations_miner = miner_predictions.get(pgt.frame_number)
        if annotations_miner is None:
            logger.info(
                f"Frame {frame_number} missing annotations_miner ({annotations_miner})"
            )
            frame_score = 0.0
        else:
            miner_bboxes_for_team1 = [
                bbox
                for bbox in annotations_miner["bboxes"]
                if bbox.cluster_id == TEAM1_SHIRT_COLOUR
            ]
            miner_bboxes_for_team2 = [
                bbox
                for bbox in annotations_miner["bboxes"]
                if bbox.cluster_id == TEAM2_SHIRT_COLOUR
            ]

            colours_in_frame = [bbox.cluster_id for bbox in pgt.annotation.bboxes]
            top_2 = Counter(colours_in_frame).most_common(2)
            logger.info(top_2)

            object_team_scores = []
            for shirt_colour, _ in top_2:
                pgt_bboxes_for_shirt_colour = [
                    bbox
                    for bbox in pgt.annotation.bboxes
                    if bbox.cluster_id == shirt_colour
                ]
                miner_iou_team1 = iou_bboxes(
                    bboxes1=pgt_bboxes_for_shirt_colour,
                    bboxes2=miner_bboxes_for_team1,
                    image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                    image_width=settings.SCOREVISION_IMAGE_WIDTH,
                )
                miner_iou_team2 = iou_bboxes(
                    bboxes1=pgt_bboxes_for_shirt_colour,
                    bboxes2=miner_bboxes_for_team2,
                    image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                    image_width=settings.SCOREVISION_IMAGE_WIDTH,
                )
                # NOTE: which team is 1 and which is 2 is decided arbitrarily by the miner, so we compare both teams and take highest
                object_team_scores.append(max(miner_iou_team1, miner_iou_team2))
            frame_score = sum(object_team_scores) / len(object_team_scores)
        logger.info(f"[compare_team_labels] Frame {pgt.frame_number}: {frame_score}")
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)


def compare_object_labels(
    pseudo_gt: list[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> float:
    settings = get_settings()
    frame_scores = []
    for pgt in pseudo_gt:
        annotations_miner = miner_predictions.get(pgt.frame_number)
        if annotations_miner is None:
            logger.info(
                f"Frame {frame_number} missing annotations_miner ({annotations_miner})"
            )
            frame_score = 0.0
        else:
            object_type_scores = []
            for object_type in ObjectOfInterest:
                pgt_bboxes_for_object_type = [
                    bbox for bbox in pgt.annotation.bboxes if bbox.label == object_type
                ]
                miner_bboxes_for_object_type = [
                    bbox
                    for bbox in annotations_miner["bboxes"]
                    if bbox.label == object_type
                ]
                object_type_score = iou_bboxes(
                    bboxes1=pgt_bboxes_for_object_type,
                    bboxes2=miner_bboxes_for_object_type,
                    image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                    image_width=settings.SCOREVISION_IMAGE_WIDTH,
                )
                object_type_scores.append(object_type_score)
            frame_score = sum(object_type_scores) / len(object_type_scores)
        logger.info(f"[compare_object_labels] Frame {pgt.frame_number}: {frame_score}")
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)


def compare_object_placement(
    pseudo_gt: list[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> float:
    settings = get_settings()
    frame_scores = []
    for pgt in pseudo_gt:
        annotations_miner = miner_predictions.get(pgt.frame_number)
        if annotations_miner is None:
            logger.info(
                f"Frame {frame_number} missing annotations_miner ({annotations_miner})"
            )
            frame_score = 0.0
        else:
            frame_score = iou_bboxes(
                bboxes1=pgt.annotation.bboxes,
                bboxes2=annotations_miner["bboxes"],
                image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                image_width=settings.SCOREVISION_IMAGE_WIDTH,
            )
        logger.info(
            f"[compare_object_placement] Frame {pgt.frame_number}: {frame_score}"
        )
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)
