from unittest.mock import patch
from types import SimpleNamespace

import pytest

from scorevision.validator.central.private_track.scoring import (
    PRIVATE_SCORING_VERSION,
    calculate_time_decay,
    find_best_match,
    frame_to_seconds,
    register_pillar_scorer,
    score_cricket_prediction_with_breakdown,
    score_predictions,
    score_predictions_for_pillar,
    score_predictions_with_breakdown,
)
from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction


_FAKE_SETTINGS = SimpleNamespace(PRIVATE_FRAME_RATE=25)


def _patch_settings():
    return patch(
        "scorevision.validator.central.private_track.scoring.get_settings",
        return_value=_FAKE_SETTINGS,
    )


def _cricket_gt() -> CricketDeliveryPrediction:
    return CricketDeliveryPrediction(
        match="Sri Lanka v India at Pallekele, 3rd T20I, 30 Jul 2024",
        matchid=34429,
        inningsid=1,
        overid=1,
        ball_in_over=1,
        ballid=1,
        xlsx_overs="1.01",
        scorecard_overs="0.1",
        kph=126.86,
        release_y=-0.492,
        release_z=1.946,
        bounce_x=8.001,
        bounce_y=-0.197,
        impact_x=1.727,
        impact_y=-0.031,
        impact_z=0.897,
        interception_distance=6.275,
        stump_y=0.017,
        stump_z=1.046,
        swing_angle=-2.402,
        deviation=1.104,
        runs=0,
        wickets=0,
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


def test_score_cricket_prediction_perfect_match():
    gt = _cricket_gt()
    score, breakdown = score_cricket_prediction_with_breakdown(gt, gt)
    assert score == 1.0
    assert all(value == 1.0 for value in breakdown.values())


def test_score_cricket_prediction_partial_match_uses_weighted_fields():
    gt = _cricket_gt()
    pred = CricketDeliveryPrediction(
        inningsid=1,
        overid=1,
        ball_in_over=1,
        kph=126.86,
        release_y=-0.492,
        release_z=1.946,
    )
    score, breakdown = score_cricket_prediction_with_breakdown(pred, gt)
    assert 0.2 < score < 0.3
    assert breakdown["inningsid"] == 1.0
    assert breakdown["bounce_x"] == 0.0


def test_score_cricket_prediction_exact_fields_normalize_strings_and_ids():
    gt = _cricket_gt()
    pred = CricketDeliveryPrediction(
        match=" sri  lanka V india at pallekele, 3rd t20i, 30 jul 2024 ",
        matchid=34429.0,
        inningsid=1.0,
        scorecard_overs="0.10",
        runs=0.0,
        wickets=0.0,
    )
    score, breakdown = score_cricket_prediction_with_breakdown(pred, gt)
    assert score > 0.0
    assert breakdown["match"] == 1.0
    assert breakdown["matchid"] == 1.0
    assert breakdown["inningsid"] == 1.0
    assert breakdown["scorecard_overs"] == 1.0
    assert breakdown["runs"] == 1.0
    assert breakdown["wickets"] == 1.0


def test_score_cricket_prediction_numeric_tolerance_decays():
    gt = _cricket_gt()
    pred = CricketDeliveryPrediction(
        kph=gt.kph + 1.5,
        release_y=gt.release_y + 0.075,
        release_z=gt.release_z + 0.075,
    )
    score, breakdown = score_cricket_prediction_with_breakdown(pred, gt)
    assert 0.0 < score < 1.0
    assert breakdown["kph"] == pytest.approx(0.5)
    assert breakdown["release_y"] == pytest.approx(0.5)
    assert breakdown["release_z"] == pytest.approx(0.5)


def test_score_cricket_prediction_strict_numeric_tolerance_boundaries():
    gt = _cricket_gt()
    pred = CricketDeliveryPrediction(
        kph=gt.kph + 3.0,
        bounce_x=gt.bounce_x + 0.25,
        stump_y=gt.stump_y + 0.12,
        deviation=gt.deviation + 2.0,
        swing_angle=gt.swing_angle + 2.0,
        stump_z=gt.stump_z + 0.12,
    )
    score, breakdown = score_cricket_prediction_with_breakdown(pred, gt)

    assert score == pytest.approx(0.0)
    assert breakdown["kph"] == 0.0
    assert breakdown["bounce_x"] == 0.0
    assert breakdown["stump_y"] == 0.0
    assert breakdown["deviation"] == 0.0
    assert breakdown["swing_angle"] == 0.0
    assert breakdown["stump_z"] == 0.0


def test_score_cricket_prediction_rough_educated_guess_scores_low():
    gt = _cricket_gt()
    pred = CricketDeliveryPrediction(
        match=gt.match,
        matchid=gt.matchid,
        inningsid=gt.inningsid,
        overid=gt.overid,
        ball_in_over=gt.ball_in_over,
        ballid=gt.ballid,
        xlsx_overs=gt.xlsx_overs,
        scorecard_overs=gt.scorecard_overs,
        kph=130.0,
        bounce_x=8.5,
        stump_y=0.16,
        deviation=-2.0,
        swing_angle=0.0,
        stump_z=0.8,
        runs=gt.runs,
        wickets=gt.wickets,
    )

    score, breakdown = score_cricket_prediction_with_breakdown(pred, gt)

    assert score < 0.2
    assert breakdown["kph"] == 0.0
    assert breakdown["bounce_x"] == 0.0
    assert breakdown["swing_angle"] == 0.0
    assert breakdown["stump_z"] == 0.0


def test_private_scoring_version_tracks_cricket_tolerance_change():
    assert PRIVATE_SCORING_VERSION == 4


def test_score_cricket_priority_metrics_outweigh_metadata_fields():
    gt = _cricket_gt()
    priority_pred = CricketDeliveryPrediction(
        kph=gt.kph,
        bounce_x=gt.bounce_x,
        stump_y=gt.stump_y,
        deviation=gt.deviation,
        swing_angle=gt.swing_angle,
        stump_z=gt.stump_z,
    )
    metadata_pred = CricketDeliveryPrediction(
        match=gt.match,
        matchid=gt.matchid,
        inningsid=gt.inningsid,
        overid=gt.overid,
        ball_in_over=gt.ball_in_over,
        ballid=gt.ballid,
        xlsx_overs=gt.xlsx_overs,
        scorecard_overs=gt.scorecard_overs,
    )

    priority_score, _ = score_cricket_prediction_with_breakdown(priority_pred, gt)
    metadata_score, _ = score_cricket_prediction_with_breakdown(metadata_pred, gt)

    assert priority_score == pytest.approx(0.74)
    assert metadata_score == pytest.approx(0.1)
    assert priority_score > metadata_score
