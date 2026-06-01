import numpy as np

from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_map50,
    compare_polygon_placement,
    compare_polygon_precision,
    compare_polygon_recall,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, ShirtColor


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


def test_polygon_metrics_merge_bbox_and_polygon_predictions():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                ),
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    polygon=[(20, 20), (30, 20), (30, 30), (20, 30)],
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                ),
            ],
        )
    ]
    miner_predictions = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ],
            "bboxes": [
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                )
            ],
        }
    }

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == 1.0


def test_polygon_metrics_select_pgt_geometry_to_match_miner_mode():
    pseudo_gt = [
        _pgt(
            1,
            [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
                ,
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    polygon=[(20, 20), (30, 20), (30, 30), (20, 30)],
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                ),
            ],
        )
    ]

    bbox_only = {
        1: {
            "bboxes": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                ),
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                )
            ]
        }
    }
    polygon_only = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                ),
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    polygon=[(20, 20), (30, 20), (30, 30), (20, 30)],
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                )
            ]
        }
    }
    mixed = {
        1: {
            "bboxes": [
                BoundingBox(
                    bbox_2d=(20, 20, 30, 30),
                    label="ball",
                    cluster_id=ShirtColor.OTHER,
                )
            ],
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ],
        }
    }

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=bbox_only) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=bbox_only) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=bbox_only) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=bbox_only) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=bbox_only) == 1.0

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=polygon_only) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=polygon_only) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=polygon_only) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=polygon_only) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=polygon_only) == 1.0

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=mixed) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=mixed) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=mixed) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=mixed) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=mixed) == 1.0
