import numpy as np
import pytest

from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.pillar_metric_registry import METRIC_REGISTRY
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_false_positive,
    compare_map50,
    compare_precision,
    compare_recall,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.geometry import (
    AnnotationGeometry,
    AnnotationGeometryType,
    Point2D,
)
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation


def _pgt(frame_number: int, boxes: list[BoundingBox]) -> PseudoGroundTruth:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    annotation = FrameAnnotation(
        annotations=boxes,
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
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="ball",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=20, y=20), Point2D(x=30, y=30)],
                    ),
                ),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "bboxes": [
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="ball",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=20, y=20), Point2D(x=30, y=30)],
                    ),
                ),
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
    assert compare_false_positive(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(1.0)


def test_detection_metrics_partial_match():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="ball",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=20, y=20), Point2D(x=30, y=30)],
                    ),
                ),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "bboxes": [
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
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
    assert compare_false_positive(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(1.0)


def test_detection_metrics_registered_for_detection_elements():
    for element_prefix in (ElementPrefix.OBJECT_DETECTION, ElementPrefix.PLAYER_DETECTION):
        assert (element_prefix, PillarName.MAP50) in METRIC_REGISTRY
        assert (element_prefix, PillarName.PRECISION) in METRIC_REGISTRY
        assert (element_prefix, PillarName.RECALL) in METRIC_REGISTRY
        assert (element_prefix, PillarName.FALSE_POSITIVE) in METRIC_REGISTRY


def test_map50_uses_detection_confidence_ranking():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=20, y=20), Point2D(x=30, y=30)],
                    ),
                ),
            ],
        )
    ]

    high_first = {
        1: {
            "bboxes": [
                BoundingBox(
                    label="player",
                    score=0.99,
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="player",
                    score=0.98,
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=50, y=50), Point2D(x=60, y=60)],
                    ),
                ),
            ]
        }
    }
    low_first = {
        1: {
            "bboxes": [
                BoundingBox(
                    label="player",
                    score=0.01,
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
                BoundingBox(
                    label="player",
                    score=0.99,
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=50, y=50), Point2D(x=60, y=60)],
                    ),
                ),
            ]
        }
    }

    assert compare_map50(pseudo_gt=pseudo_gt, miner_predictions=high_first) > compare_map50(
        pseudo_gt=pseudo_gt, miner_predictions=low_first
    )


def test_false_positive_uses_ffpi_formula():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(
                    label="player",
                    geometry=AnnotationGeometry(
                        type=AnnotationGeometryType.BBOX,
                        points=[Point2D(x=0, y=0), Point2D(x=10, y=10)],
                    ),
                ),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "bboxes": [
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=0, y=0), Point2D(x=10, y=10)])),
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=20, y=20), Point2D(x=30, y=30)])),
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=40, y=40), Point2D(x=50, y=50)])),
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=60, y=60), Point2D(x=70, y=70)])),
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=80, y=80), Point2D(x=90, y=90)])),
                BoundingBox(label="player", geometry=AnnotationGeometry(type=AnnotationGeometryType.BBOX, points=[Point2D(x=100, y=100), Point2D(x=110, y=110)])),
            ]
        }
    }

    # 1 TP + 5 FP over 1 image => ffpi=5 => max(0, 1 - ffpi/10) = 0.5
    assert compare_false_positive(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    ) == pytest.approx(0.5)
