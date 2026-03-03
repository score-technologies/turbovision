import numpy as np
import pytest

from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.pillar_metric_registry import METRIC_REGISTRY
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_map50,
    compare_precision,
    compare_recall,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation


def _pgt(frame_number: int, boxes: list[BoundingBox]) -> PseudoGroundTruth:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    annotation = FrameAnnotation(
        bboxes=boxes,
        category=Action.NONE,
        confidence=100,
        reason="test",
    )
    return PseudoGroundTruth(
        video_name="test.mp4",
        frame_number=frame_number,
        spatial_image=image,
        temporal_image=image,
        annotation=annotation,
    )


def test_detection_metrics_perfect_match():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(bbox_2d=(0, 0, 10, 10), label="player"),
                BoundingBox(bbox_2d=(20, 20, 30, 30), label="ball"),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "bboxes": [
                BoundingBox(bbox_2d=(0, 0, 10, 10), label="player"),
                BoundingBox(bbox_2d=(20, 20, 30, 30), label="ball"),
            ]
        }
    }

    assert compare_map50(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(1.0)
    assert (
        compare_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)
        == pytest.approx(1.0)
    )
    assert compare_recall(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(1.0)


def test_detection_metrics_partial_match():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(bbox_2d=(0, 0, 10, 10), label="player"),
                BoundingBox(bbox_2d=(20, 20, 30, 30), label="ball"),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "bboxes": [
                BoundingBox(bbox_2d=(0, 0, 10, 10), label="player"),
            ]
        }
    }

    assert compare_map50(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(0.5)
    assert (
        compare_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)
        == pytest.approx(1.0)
    )
    assert compare_recall(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(0.5)


def test_detection_metrics_registered_for_detection_elements():
    for element_prefix in (ElementPrefix.OBJECT_DETECTION, ElementPrefix.PLAYER_DETECTION):
        assert (element_prefix, PillarName.MAP50) in METRIC_REGISTRY
        assert (element_prefix, PillarName.PRECISION) in METRIC_REGISTRY
        assert (element_prefix, PillarName.RECALL) in METRIC_REGISTRY
