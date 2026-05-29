import pytest

from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_false_positive,
    compare_map50,
    compare_object_counts,
    compare_object_placement,
    compare_precision,
    compare_recall,
)
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_false_positive,
    compare_polygon_map50,
    compare_polygon_placement,
    compare_polygon_precision,
    compare_polygon_recall,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import (
    BoundingBox,
    FrameAnnotation,
    ShirtColor,
)


def _rect_polygon(box: BoundingBox) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = box.bbox_2d
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _polygon_box(box: BoundingBox) -> BoundingBox:
    return BoundingBox(
        bbox_2d=box.bbox_2d,
        polygon=_rect_polygon(box),
        label=box.label,
        score=box.score,
        cluster_id=box.cluster_id,
    )


def _triangular_polygon_box(box: BoundingBox) -> BoundingBox:
    x1, y1, x2, y2 = box.bbox_2d
    return BoundingBox(
        bbox_2d=box.bbox_2d,
        polygon=[
            (x1, y1),
            (x2, y1),
            (x1 + 1, y1 + 1),
        ],
        label=box.label,
        score=box.score,
        cluster_id=box.cluster_id,
    )


def _fixture() -> tuple[list[PseudoGroundTruth], dict[int, dict], dict[int, dict]]:
    gt_boxes = [
        BoundingBox(
            bbox_2d=(0, 0, 10, 10),
            label="player",
            cluster_id=ShirtColor.WHITE,
        ),
        BoundingBox(
            bbox_2d=(20, 20, 30, 30),
            label="ball",
            cluster_id=ShirtColor.OTHER,
        ),
    ]
    pred_boxes = [
        BoundingBox(
            bbox_2d=(0, 0, 10, 10),
            label="player",
            score=0.9,
            cluster_id=ShirtColor.WHITE,
        ),
        BoundingBox(
            bbox_2d=(20, 20, 30, 30),
            label="ball",
            score=0.8,
            cluster_id=ShirtColor.OTHER,
        ),
    ]
    pseudo_gt = [
        PseudoGroundTruth(
            video_name="v",
            frame_number=1,
            spatial_image=None,
            temporal_image=None,
            annotation=FrameAnnotation(
                bboxes=[_polygon_box(box) for box in gt_boxes],
                category=Action.NONE,
                confidence=100,
                reason="test",
            ),
        )
    ]
    bbox_miner = {1: {"bboxes": pred_boxes}}
    polygon_miner = {1: {"polygons": [_polygon_box(box) for box in pred_boxes]}}
    return pseudo_gt, bbox_miner, polygon_miner


def test_polygon_metrics_match_bbox_metrics_on_rectangular_geometry():
    pseudo_gt, bbox_miner, polygon_miner = _fixture()

    metric_pairs = [
        (compare_object_placement, compare_polygon_placement),
        (compare_map50, compare_polygon_map50),
        (compare_object_counts, compare_polygon_counts),
        (compare_precision, compare_polygon_precision),
        (compare_recall, compare_polygon_recall),
        (compare_false_positive, compare_polygon_false_positive),
    ]

    for bbox_metric, polygon_metric in metric_pairs:
        bbox_score = bbox_metric(pseudo_gt=pseudo_gt, miner_predictions=bbox_miner)
        polygon_score = polygon_metric(
            pseudo_gt=pseudo_gt, miner_predictions=polygon_miner
        )
        assert polygon_score == pytest.approx(bbox_score)


def test_polygon_metrics_diverge_on_non_rectangular_geometry():
    gt_boxes = [
        BoundingBox(
            bbox_2d=(0, 0, 10, 10),
            label="player",
            cluster_id=ShirtColor.WHITE,
        ),
        BoundingBox(
            bbox_2d=(20, 20, 30, 30),
            label="ball",
            cluster_id=ShirtColor.OTHER,
        ),
    ]
    pred_boxes = [
        BoundingBox(
            bbox_2d=(0, 0, 10, 10),
            label="player",
            score=0.9,
            cluster_id=ShirtColor.WHITE,
        ),
        BoundingBox(
            bbox_2d=(20, 20, 30, 30),
            label="ball",
            score=0.8,
            cluster_id=ShirtColor.OTHER,
        ),
    ]
    pseudo_gt = [
        PseudoGroundTruth(
            video_name="v",
            frame_number=1,
            spatial_image=None,
            temporal_image=None,
            annotation=FrameAnnotation(
                bboxes=[_polygon_box(box) for box in gt_boxes],
                category=Action.NONE,
                confidence=100,
                reason="test",
            ),
        )
    ]
    bbox_miner = {1: {"bboxes": pred_boxes}}

    irregular_polygon_miner = {
        1: {
            "polygons": [
                _triangular_polygon_box(box)
                for box in bbox_miner[1]["bboxes"]
            ]
        }
    }

    metric_pairs = [
        (compare_object_placement, compare_polygon_placement),
        (compare_map50, compare_polygon_map50),
        (compare_object_counts, compare_polygon_counts),
        (compare_precision, compare_polygon_precision),
        (compare_recall, compare_polygon_recall),
        (compare_false_positive, compare_polygon_false_positive),
    ]

    for bbox_metric, polygon_metric in metric_pairs:
        bbox_score = bbox_metric(pseudo_gt=pseudo_gt, miner_predictions=bbox_miner)
        polygon_score = polygon_metric(
            pseudo_gt=pseudo_gt, miner_predictions=irregular_polygon_miner
        )
        assert polygon_score < bbox_score
