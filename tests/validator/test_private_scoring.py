from unittest.mock import patch
from types import SimpleNamespace

from scorevision.validator.central.private_track.scoring import (
    calculate_time_decay,
    find_best_match,
    frame_to_seconds,
    register_pillar_scorer,
    score_predictions,
    score_predictions_for_pillar,
    score_predictions_with_breakdown,
)
from scorevision.utils.schemas import FramePrediction


_FAKE_SETTINGS = SimpleNamespace(PRIVATE_FRAME_RATE=25)


def _patch_settings():
    return patch(
        "scorevision.validator.central.private_track.scoring.get_settings",
        return_value=_FAKE_SETTINGS,
    )


def test_frame_to_seconds():
    with _patch_settings():
        assert frame_to_seconds(0) == 0.0
        assert frame_to_seconds(25) == 1.0
        assert frame_to_seconds(50) == 2.0


def test_calculate_time_decay_within_tolerance():
    assert calculate_time_decay(0.0, 1.0, 0.0) == 1.0
    assert calculate_time_decay(0.5, 1.0, 0.0) == 0.5
    assert calculate_time_decay(1.0, 1.0, 0.0) == 0.0


def test_calculate_time_decay_beyond_tolerance():
    assert calculate_time_decay(2.0, 1.0, 0.0) == 0.0


def test_calculate_time_decay_with_min_score():
    result = calculate_time_decay(0.5, 1.0, 0.5)
    assert 0.5 < result < 1.0


def test_score_predictions_empty_ground_truth():
    with _patch_settings():
        assert score_predictions([], []) == 0.0


def test_score_predictions_perfect_match():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [FramePrediction(frame=25, action="pass")]
        score = score_predictions(preds, gt)
        assert score == 1.0


def test_score_predictions_no_predictions():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        score = score_predictions([], gt)
        assert score == 0.0


def test_score_predictions_unmatched_penalty():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [
            FramePrediction(frame=25, action="pass"),
            FramePrediction(frame=100, action="pass"),
        ]
        score = score_predictions(preds, gt)
        assert score == 0.0


def test_score_predictions_with_private_legacy_only():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [FramePrediction(frame=25, action="pass")]
        score = score_predictions(preds, gt, pillar_weights={"soccer_action": 1.0})
        assert score == 1.0


def test_score_predictions_with_private_legacy_pillar_matches_legacy_value():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [
            FramePrediction(frame=25, action="pass"),
            FramePrediction(frame=200, action="goal"),
        ]
        legacy = score_predictions(preds, gt)
        score, breakdown = score_predictions_with_breakdown(
            preds,
            gt,
            pillar_weights={"soccer_action": 1.0},
        )
        assert score == legacy
        assert breakdown["soccer_action"] == legacy


def test_score_predictions_with_unsupported_pillar_returns_zero_weighted_score():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [FramePrediction(frame=25, action="pass")]
        score, breakdown = score_predictions_with_breakdown(
            preds,
            gt,
            pillar_weights={"role": 1.0},
        )
        assert score == 0.0
        assert breakdown["role"] == 0.0


def test_score_predictions_for_pillar_soccer_action_matches_legacy():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [FramePrediction(frame=25, action="pass")]
        legacy = score_predictions(preds, gt)
        pillar_score = score_predictions_for_pillar(
            pillar="soccer_action",
            predictions=preds,
            ground_truth=gt,
        )
        assert pillar_score == legacy


def test_register_pillar_scorer_dispatches_custom_pillar():
    with _patch_settings():
        gt = [FramePrediction(frame=25, action="pass")]
        preds = [FramePrediction(frame=25, action="pass")]

        def rugby_scorer(
            predictions: list[FramePrediction],
            ground_truth: list[FramePrediction],
        ) -> float:
            assert predictions == preds
            assert ground_truth == gt
            return 0.42

        register_pillar_scorer("rugby_action", rugby_scorer)

        score, breakdown = score_predictions_with_breakdown(
            preds,
            gt,
            pillar_weights={"rugby_action": 1.0},
        )
        assert score == 0.42
        assert breakdown["rugby_action"] == 0.42
