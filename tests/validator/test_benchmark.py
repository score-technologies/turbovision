import pytest
from unittest.mock import patch, MagicMock
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.actions import ACTION_CLASS_INDEX
from scorevision.validator.central.private_track.benchmark import (
    compute_map_at_1s,
    _vectorize_ground_truth,
    _vectorize_predictions,
)
import numpy as np


def _mock_settings():
    s = MagicMock()
    s.PRIVATE_FRAME_RATE = 25
    s.BENCHMARK_PRECISION_RECALL_THRESHOLDS = 200
    s.BENCHMARK_AP_INTERPOLATION_POINTS = 11
    s.BENCHMARK_MAX_VIDEO_DURATION_MINUTES = 120
    return s


@pytest.fixture(autouse=True)
def _patch_settings():
    with patch(
        "scorevision.validator.central.private_track.benchmark.get_settings",
        return_value=_mock_settings(),
    ):
        yield


def test_perfect_predictions_yield_map_of_one():
    ground_truth = [
        FramePrediction(frame=100, action="pass"),
        FramePrediction(frame=200, action="goal"),
    ]
    predictions = [
        FramePrediction(frame=100, action="pass"),
        FramePrediction(frame=200, action="goal"),
    ]
    result = compute_map_at_1s(predictions, ground_truth)
    assert result.per_action_ap["pass"] == 1.0
    assert result.per_action_ap["goal"] == 1.0


def test_no_predictions_yield_map_of_zero():
    ground_truth = [
        FramePrediction(frame=100, action="pass"),
        FramePrediction(frame=200, action="goal"),
    ]
    result = compute_map_at_1s([], ground_truth)
    assert result.map_at_1s == 0.0


def test_completely_wrong_actions_yield_zero_for_those_classes():
    ground_truth = [FramePrediction(frame=100, action="pass")]
    predictions = [FramePrediction(frame=100, action="goal")]
    result = compute_map_at_1s(predictions, ground_truth)
    assert result.per_action_ap["pass"] == 0.0


def test_prediction_within_tolerance_is_matched():
    ground_truth = [FramePrediction(frame=100, action="pass")]
    predictions = [FramePrediction(frame=112, action="pass")]
    result = compute_map_at_1s(predictions, ground_truth)
    assert result.per_action_ap["pass"] > 0.0


def test_prediction_outside_tolerance_is_not_matched():
    ground_truth = [FramePrediction(frame=100, action="pass")]
    predictions = [FramePrediction(frame=200, action="pass")]
    result = compute_map_at_1s(predictions, ground_truth)
    assert result.per_action_ap["pass"] == 0.0


def test_vectorize_ground_truth_places_value_at_correct_index():
    gt = [FramePrediction(frame=50, action="pass")]
    vector = _vectorize_ground_truth(gt, 1000)
    class_idx = ACTION_CLASS_INDEX["pass"]
    assert vector[50, class_idx] == 1
    assert vector[51, class_idx] == 0


def test_vectorize_predictions_places_confidence_at_correct_index():
    preds = [FramePrediction(frame=50, action="goal", confidence=1.0)]
    vector = _vectorize_predictions(preds, 1000)
    class_idx = ACTION_CLASS_INDEX["goal"]
    assert vector[50, class_idx] == 1.0
    assert vector[49, class_idx] == -1


def test_unknown_action_is_ignored():
    gt = [FramePrediction(frame=50, action="nonexistent_action")]
    vector = _vectorize_ground_truth(gt, 1000)
    assert np.sum(vector) == 0.0


def test_result_contains_all_action_classes():
    ground_truth = [FramePrediction(frame=100, action="pass")]
    predictions = [FramePrediction(frame=100, action="pass")]
    result = compute_map_at_1s(predictions, ground_truth)
    for action_name in ACTION_CLASS_INDEX:
        assert action_name in result.per_action_ap


def test_empty_ground_truth_yields_zero():
    predictions = [FramePrediction(frame=100, action="pass")]
    result = compute_map_at_1s(predictions, [])
    assert result.map_at_1s == 0.0


def test_multiple_predictions_for_same_gt_only_one_matches():
    ground_truth = [FramePrediction(frame=100, action="pass")]
    predictions = [
        FramePrediction(frame=99, action="pass"),
        FramePrediction(frame=100, action="pass"),
        FramePrediction(frame=101, action="pass"),
    ]
    result = compute_map_at_1s(predictions, ground_truth)
    assert result.per_action_ap["pass"] > 0.0
    assert result.per_action_ap["pass"] < 1.0
