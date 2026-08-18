import math
import random

import numpy as np
import pytest

from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_false_positive,
    compare_map50,
    compare_precision,
    compare_recall,
)
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    ADAPTIVE_IOU_ANCHORS,
    _adaptive_iou_threshold_for_gt,
    _gt_area_ratio,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation


def _pgt(
    frame_number: int, boxes: list[BoundingBox], *, image_shape: tuple[int, int] = (1000, 1000)
) -> PseudoGroundTruth:
    height, width = image_shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    annotation = FrameAnnotation(
        bboxes=boxes, category=Action.NONE, confidence=100, reason="test"
    )
    return PseudoGroundTruth(
        video_name="test.mp4",
        frame_number=frame_number,
        spatial_image=image,
        temporal_image=image,
        annotation=annotation,
    )


def _gt_box(w: float, h: float, *, label: str = "player") -> BoundingBox:
    return BoundingBox(bbox_2d=(0, 0, round(w), round(h)), label=label)


# ===============================
# Unit tests: threshold formula itself
# ===============================


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.0005, 0.30),  # r_min anchor
        (0.01, 0.50),  # start of plateau
        (0.02, 0.50),  # inside plateau
        (0.03, 0.50),  # inside plateau
        (0.05, 0.50),  # end of plateau
        (0.25, 0.70),  # r_max anchor
    ],
)
def test_adaptive_threshold_matches_anchor_points(ratio, expected):
    gt = _gt_box(1, 1)  # geometry irrelevant, ratio forced via image size below
    # image_height=1 keeps gt_area/image_area == ratio exact (no sqrt/rounding drift).
    image_width = round(1 / ratio)
    threshold = _adaptive_iou_threshold_for_gt(gt, image_height=1, image_width=image_width)
    assert threshold == pytest.approx(expected, abs=0.01)


def test_adaptive_threshold_clamped_below_r_min():
    gt = _gt_box(1, 1)
    threshold = _adaptive_iou_threshold_for_gt(gt, image_height=100_000, image_width=100_000)
    assert threshold == pytest.approx(ADAPTIVE_IOU_ANCHORS[0][1])


def test_adaptive_threshold_clamped_above_r_max():
    gt = _gt_box(900, 900)
    threshold = _adaptive_iou_threshold_for_gt(gt, image_height=1000, image_width=1000)
    assert threshold == pytest.approx(ADAPTIVE_IOU_ANCHORS[-1][1])


def test_adaptive_threshold_monotonic_non_decreasing_with_size():
    rng = random.Random(42)
    ratios = sorted(10 ** rng.uniform(-5, -0.3) for _ in range(500))
    thresholds = []
    gt = _gt_box(1000, 1)
    for ratio in ratios:
        image_width = round(1000 / ratio)
        thresholds.append(
            _adaptive_iou_threshold_for_gt(gt, image_height=1, image_width=image_width)
        )
    assert all(t2 >= t1 - 1e-9 for t1, t2 in zip(thresholds, thresholds[1:]))
    assert min(thresholds) >= ADAPTIVE_IOU_ANCHORS[0][1] - 1e-9
    assert max(thresholds) <= ADAPTIVE_IOU_ANCHORS[-1][1] + 1e-9


def test_gt_area_ratio_uses_settings_default_when_image_size_missing():
    gt = _gt_box(96, 54)  # 5184 px
    ratio = _gt_area_ratio(gt, image_height=None, image_width=None)
    # default settings: 960x540 -> area 518400
    assert ratio == pytest.approx(5184 / 518400)


# ===============================
# Integration / stress tests: does it actually change TP/FP outcomes?
# ===============================


def test_small_object_below_old_fixed_threshold_now_counts_as_tp():
    """
    GT is a 32x32 box on a 1000x1000 image -> ratio ~0.10% -> required IoU ~0.35.
    Prediction is offset by (8, 8) -> IoU ~0.39.
    Under the old fixed 0.5 threshold this would have been a miss (FP); with the
    adaptive threshold it should now be accepted as a match (TP).
    """
    pseudo_gt = [_pgt(1, [_gt_box(32, 32)])]
    pred = BoundingBox(bbox_2d=(8, 8, 40, 40), label="player")
    miner_predictions = {1: {"bboxes": [pred]}}

    required = _adaptive_iou_threshold_for_gt(
        pseudo_gt[0].annotation.bboxes[0], image_height=1000, image_width=1000
    )
    assert required < 0.5  # sanity: bar really did drop below the old fixed value

    assert compare_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)
    assert compare_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)
    assert compare_recall(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)
    assert compare_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)


def test_small_object_below_adaptive_threshold_still_counts_as_fp():
    """
    Same 32x32 GT (required IoU ~0.35) but prediction offset by (10, 10) -> IoU ~0.31,
    which is still below the (lower) required threshold, so it must still be rejected.
    """
    pseudo_gt = [_pgt(1, [_gt_box(32, 32)])]
    pred = BoundingBox(bbox_2d=(10, 10, 42, 42), label="player")
    miner_predictions = {1: {"bboxes": [pred]}}

    required = _adaptive_iou_threshold_for_gt(
        pseudo_gt[0].annotation.bboxes[0], image_height=1000, image_width=1000
    )

    assert compare_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.0)
    assert compare_recall(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.0)
    # false_positive pillar: 1 FP over 1 image -> ffpi=1 -> score = 1 - 1/10 = 0.9
    assert compare_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.9)
    assert required < 0.5  # confirms the rejection is NOT an artifact of a stricter bar


def test_large_object_above_old_fixed_threshold_now_counts_as_fp():
    """
    GT is a 600x500 box on a 1000x1000 image -> ratio 30% (clamped to r_max) -> required IoU 0.70.
    Prediction offset by (150, 0) -> IoU exactly 0.6.
    Under the old fixed 0.5 threshold this would have passed; with the adaptive
    threshold (0.70 for large objects) it must now be rejected.
    """
    pseudo_gt = [_pgt(1, [_gt_box(600, 500)])]
    pred = BoundingBox(bbox_2d=(150, 0, 750, 500), label="player")
    miner_predictions = {1: {"bboxes": [pred]}}

    required = _adaptive_iou_threshold_for_gt(
        pseudo_gt[0].annotation.bboxes[0], image_height=1000, image_width=1000
    )
    assert required == pytest.approx(0.70)

    assert compare_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.0)
    assert compare_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.0)
    assert compare_recall(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(0.0)


def test_large_object_above_adaptive_threshold_counts_as_tp():
    """Same 600x500 GT, but shifted by only (50, 0) -> IoU well above 0.70 -> must be a TP."""
    pseudo_gt = [_pgt(1, [_gt_box(600, 500)])]
    pred = BoundingBox(bbox_2d=(50, 0, 650, 500), label="player")
    miner_predictions = {1: {"bboxes": [pred]}}

    assert compare_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)
    assert compare_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions) == pytest.approx(1.0)


def test_plateau_gives_identical_threshold_for_different_mid_size_objects():
    """Ratios of 1.5% and 4% both fall in the flat 1%-5% plateau -> same required IoU (0.50)."""
    image_area = 1_000_000
    gt_1_5pct = _gt_box(*_square_side_for_ratio(0.015, image_area))
    gt_4pct = _gt_box(*_square_side_for_ratio(0.04, image_area))

    t1 = _adaptive_iou_threshold_for_gt(gt_1_5pct, image_height=1000, image_width=1000)
    t2 = _adaptive_iou_threshold_for_gt(gt_4pct, image_height=1000, image_width=1000)
    assert t1 == pytest.approx(0.50)
    assert t2 == pytest.approx(0.50)


def _square_side_for_ratio(ratio: float, image_area: float) -> tuple[float, float]:
    side = round(math.sqrt(ratio * image_area))
    return side, side


def test_stress_many_frames_mixed_object_sizes_stay_within_bounds():
    """
    Stress test: many frames, each with a mix of tiny/mid/huge GT objects and
    predictions at varying offsets. The pipeline must not crash and every
    resulting score must stay within [0, 1].
    """
    rng = random.Random(7)
    pseudo_gt = []
    miner_predictions: dict[int, dict] = {}

    size_choices = [(10, 10), (32, 32), (100, 100), (300, 300), (700, 700)]

    for frame in range(200):
        n_objects = rng.randint(1, 5)
        gt_boxes = []
        pred_boxes = []
        for i in range(n_objects):
            w, h = rng.choice(size_choices)
            x0, y0 = round(rng.uniform(0, 900)), round(rng.uniform(0, 900))
            x1, y1 = min(1000, x0 + w), min(1000, y0 + h)
            gt_boxes.append(BoundingBox(bbox_2d=(x0, y0, x1, y1), label="player"))

            # jitter the prediction, occasionally drop it or add a spurious extra one
            if rng.random() < 0.85:
                jitter = round(rng.uniform(-w * 0.4, w * 0.4))
                px0, py0 = max(0, x0 + jitter), max(0, y0 + jitter)
                pred_boxes.append(
                    BoundingBox(
                        bbox_2d=(px0, py0, px0 + w, py0 + h),
                        label="player",
                        score=rng.uniform(0.1, 1.0),
                    )
                )
            if rng.random() < 0.1:
                fx, fy = round(rng.uniform(0, 950)), round(rng.uniform(0, 950))
                pred_boxes.append(
                    BoundingBox(
                        bbox_2d=(fx, fy, fx + 20, fy + 20),
                        label="player",
                        score=rng.uniform(0.1, 1.0),
                    )
                )

        pseudo_gt.append(_pgt(frame, gt_boxes))
        miner_predictions[frame] = {"bboxes": pred_boxes}

    map50 = compare_map50(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)
    precision = compare_precision(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)
    recall = compare_recall(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)
    false_positive = compare_false_positive(pseudo_gt=pseudo_gt, miner_predictions=miner_predictions)

    for value in (map50, precision, recall, false_positive):
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)
