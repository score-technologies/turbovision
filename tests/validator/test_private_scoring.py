from unittest.mock import patch
from types import SimpleNamespace
import pytest

from scorevision.validator.central.private_track.scoring import (
    calculate_time_decay,
    find_best_match,
    frame_to_seconds,
    register_pillar_scorer,
    score_cricket_prediction_with_breakdown,
    score_snooker_ball_state_with_breakdown,
    score_predictions,
    score_predictions_for_pillar,
    score_predictions_with_breakdown,
)
from scorevision.utils.schemas import (
    CricketDeliveryPrediction,
    FramePrediction,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)


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


def test_cricket_scoring_top6_fields_perfect_match():
    prediction = CricketDeliveryPrediction(
        kph=130.0,
        bounce_x=6.0,
        stump_y=0.2,
        deviation=1.0,
        swing_angle=-0.5,
        stump_z=0.8,
    )
    score, breakdown = score_cricket_prediction_with_breakdown(prediction, prediction)

    # Only the 6 heaviest-weight fields are present in this fixture.
    # Their total weight is 0.74, so a perfect match on those fields yields 0.74.
    assert score == pytest.approx(0.74)
    assert breakdown["kph"] == 1.0


def test_snooker_ball_state_perfect_match():
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=0,
                balls=[
                    SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.6, y=0.6, state="on_table"),
                ],
            )
        ]
    )
    score, breakdown = score_snooker_ball_state_with_breakdown(prediction, prediction)

    assert score == pytest.approx(1.0)
    assert breakdown["snooker_ball_state"] == pytest.approx(1.0)


def test_snooker_ball_state_missing_target_frame_scores_zero():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=999,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(0.0)
    assert breakdown["snooker_ball_state"] == pytest.approx(0.0)


def test_snooker_ball_state_empty_prediction_scores_zero():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(frames=[SnookerBallStateFrame(frame=50, balls=[])])

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(0.0)
    assert breakdown["false_positive_score"] == pytest.approx(0.0)


def test_snooker_ball_state_ignores_extra_non_target_frames():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            ),
            SnookerBallStateFrame(
                frame=999,
                balls=[SnookerBallPrediction(label="invalid", x=0.1, y=0.1, state="on_table")],
            ),
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(1.0)
    assert breakdown["snooker_ball_state"] == pytest.approx(1.0)


def test_snooker_ball_state_matches_reds_by_hungarian_assignment():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="red", x=0.2, y=0.2, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.8, y=0.8, state="on_table"),
                ],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="red", x=0.8, y=0.8, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.2, y=0.2, state="on_table"),
                ],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(1.0)
    assert breakdown["coordinate_accuracy"] == pytest.approx(1.0)


def test_snooker_ball_state_penalizes_wrong_red_count():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=0,
                balls=[
                    SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.6, y=0.6, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.7, y=0.7, state="on_table"),
                ],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=0,
                balls=[
                    SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.6, y=0.6, state="on_table"),
                ],
            )
        ]
    )
    score, breakdown = score_snooker_ball_state_with_breakdown(prediction, ground_truth)

    assert score < 1.0
    assert breakdown["red_count_accuracy"] < 1.0


def test_snooker_ball_state_penalizes_duplicate_unique_colour():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="blue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="blue", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="blue", x=0.6, y=0.6, state="on_table"),
                ],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score < 1.0
    assert breakdown["false_positive_score"] < 1.0


def test_snooker_ball_state_allows_potted_and_occluded_without_coordinates():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="cue", state="potted"),
                    SnookerBallPrediction(label="blue", state="occluded"),
                ],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="cue", state="potted"),
                    SnookerBallPrediction(label="blue", state="occluded"),
                ],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(1.0)
    assert breakdown["state_accuracy"] == pytest.approx(1.0)


def test_snooker_ball_state_penalizes_invalid_labels_and_missing_on_table_coordinates():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="orange", x=0.5, y=0.5, state="on_table"),
                    SnookerBallPrediction(label="cue", state="on_table"),
                ],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score < 1.0
    assert breakdown["identity_accuracy"] == pytest.approx(0.0)
    assert breakdown["false_positive_score"] == pytest.approx(0.0)


def test_snooker_ball_state_uses_tighter_coordinate_tolerance():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.50, y=0.50, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.56, y=0.50, state="on_table")],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(0.0)
    assert breakdown["coordinate_accuracy"] == pytest.approx(0.0)
    assert breakdown["identity_accuracy"] == pytest.approx(0.0)


def test_snooker_ball_state_does_not_normalize_invalid_state_to_unknown():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="blue", state="unknown")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="blue", state="hidden")],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(0.0)
    assert breakdown["identity_accuracy"] == pytest.approx(0.0)
    assert breakdown["false_positive_score"] == pytest.approx(0.0)


def test_snooker_ball_state_invalid_reds_do_not_earn_red_count_credit():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="red", x=0.30, y=0.30, state="on_table"),
                    SnookerBallPrediction(label="red", x=0.40, y=0.40, state="on_table"),
                ],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[
                    SnookerBallPrediction(label="red", state="on_table"),
                    SnookerBallPrediction(label="red", state="on_table"),
                ],
            )
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score == pytest.approx(0.0)
    assert breakdown["red_count_accuracy"] == pytest.approx(0.0)
    assert breakdown["false_positive_score"] == pytest.approx(0.0)


def test_snooker_ball_state_penalizes_duplicate_target_frame_objects():
    ground_truth = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            )
        ]
    )
    prediction = SnookerBallStatePrediction(
        frames=[
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            ),
            SnookerBallStateFrame(
                frame=50,
                balls=[SnookerBallPrediction(label="cue", x=0.5, y=0.5, state="on_table")],
            ),
        ]
    )

    score, breakdown = score_snooker_ball_state_with_breakdown(
        prediction,
        ground_truth,
        target_frames=[50],
    )

    assert score < 1.0
    assert breakdown["false_positive_score"] < 1.0
    assert breakdown["snooker_ball_state"] == pytest.approx(score)
