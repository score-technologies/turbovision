from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_false_positive,
    compare_polygon_placement,
    compare_polygon_map50,
    compare_polygon_precision,
    compare_polygon_recall,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, ShirtColor


def test_polygon_metrics_registered():
    assert True


def test_polygon_placeholder_metrics_return_zero():
    gt = PseudoGroundTruth(
        video_name="v",
        frame_number=1,
        spatial_image=None,  # not used by the scoring functions
        temporal_image=None,
        annotation=FrameAnnotation(
            bboxes=[
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ],
            category=Action.NONE,
            confidence=100,
            reason="test",
        ),
    )
    miner = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ]
        }
    }
    pseudo_gt = [gt]

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0


def test_polygon_metrics_accept_bbox_only_miner_inputs():
    gt = PseudoGroundTruth(
        video_name="v",
        frame_number=1,
        spatial_image=None,
        temporal_image=None,
        annotation=FrameAnnotation(
            bboxes=[
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ],
            category=Action.NONE,
            confidence=100,
            reason="test",
        ),
    )
    miner = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    label="player",
                    cluster_id=ShirtColor.WHITE,
                )
            ]
        }
    }

    pseudo_gt = [gt]

    assert compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
    assert compare_polygon_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner) == 1.0
