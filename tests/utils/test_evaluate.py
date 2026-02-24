from scorevision.utils.evaluate import (
    post_vlm_ranking,
    get_element_scores,
    parse_miner_prediction,
)
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)
from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.data_models import SVRunOutput
from scorevision.vlm_pipeline.utils.response_models import (
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)

from pytest import raises


def test_post_vlm_ranking(
    dummy_manifest,
    dummy_pseudo_gt_annotations,
    fake_miner_predictions,
    fake_payload,
    fake_challenge,
    fake_frame_store,
) -> None:
    evaluation = post_vlm_ranking(
        payload=fake_payload,
        miner_run=fake_miner_predictions,
        challenge=fake_challenge,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        frame_store=fake_frame_store,
        manifest=dummy_manifest,
    )
    assert isinstance(evaluation.acc_breakdown, dict)
    assert isinstance(evaluation.details, dict)
    assert evaluation.latency_ms == 0.0
    assert evaluation.acc > 0.0


def test_get_element_scores(
    dummy_manifest,
    dummy_pseudo_gt_annotations,
    fake_frame_store,
    fake_miner_predictions,
) -> None:
    breakdown = get_element_scores(
        manifest=dummy_manifest,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        miner_run=fake_miner_predictions,
        frame_store=fake_frame_store,
        challenge_type=ChallengeType.FOOTBALL,
    )
    assert isinstance(breakdown, dict)
    assert "mean_weighted" in breakdown
    assert breakdown["mean_weighted"] > 0.0


def test_get_element_scores_one_pillar_with_zero_weight(
    manifest_with_pillar_weight_of_zero,
    dummy_pseudo_gt_annotations,
    fake_frame_store,
    fake_miner_predictions,
):
    breakdown = get_element_scores(
        manifest=manifest_with_pillar_weight_of_zero,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        miner_run=fake_miner_predictions,
        frame_store=fake_frame_store,
        challenge_type=ChallengeType.FOOTBALL,
    )
    assert (
        breakdown[ElementPrefix.PLAYER_DETECTION.value][PillarName.COUNT.value][
            "weighted_score"
        ]
        == 0.0
    )  # the element pillar with a weight of 0.0 should produce a 0.0 weighted score
    assert (
        breakdown[ElementPrefix.PLAYER_DETECTION.value][PillarName.IOU.value][
            "weighted_score"
        ]
        > 0.0
    )  # the element pillar with a non-zero weight
    assert breakdown["mean_weighted"] > 0.0


def test_get_element_scores_on_pillar_without_metric_raises_error(
    manifest_with_pillar_that_has_no_metric_registered,
    dummy_pseudo_gt_annotations,
    fake_frame_store,
    fake_miner_predictions,
):
    with raises(NotImplementedError) as exc_info:
        get_element_scores(
            manifest=manifest_with_pillar_that_has_no_metric_registered,
            pseudo_gt_annotations=dummy_pseudo_gt_annotations,
            miner_run=fake_miner_predictions,
            frame_store=fake_frame_store,
            challenge_type=ChallengeType.FOOTBALL,
        )


def test_get_element_scores_economics(
    dummy_manifest,
    dummy_pseudo_gt_annotations,
    fake_miner_predictions,
    fake_frame_store,
):
    scores = get_element_scores(
        manifest=dummy_manifest,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        miner_run=fake_miner_predictions,
        frame_store=fake_frame_store,
        challenge_type=ChallengeType.FOOTBALL,
    )
    for element in dummy_manifest.elements:
        gated_score = scores[element.category]["total_weighted_and_gated"]
        assert gated_score >= element.delta_floor or gated_score == 0.0


def test_parse_miner_prediction_maps_cluster_id_for_team_scoring():
    miner_run = SVRunOutput(
        success=True,
        latency_ms=0.0,
        predictions={
            "frames": [
                {
                    "frame_id": 12,
                    "boxes": [
                        {
                            "x1": 1,
                            "y1": 2,
                            "x2": 10,
                            "y2": 20,
                            "cls_id": 0,
                            "cluster_id": "team1",
                        },
                        {
                            "x1": 5,
                            "y1": 6,
                            "x2": 15,
                            "y2": 26,
                            "cls_id": 0,
                            "team_id": 1,
                        },
                        {
                            "x1": 11,
                            "y1": 12,
                            "x2": 20,
                            "y2": 30,
                            "cls_id": 0,
                            "team_id": 2,
                        },
                    ],
                    "keypoints": [],
                }
            ]
        },
        error=None,
        model=None,
    )

    parsed = parse_miner_prediction(miner_run=miner_run, object_names=["player"])
    bboxes = parsed[12]["bboxes"]

    assert bboxes[0].cluster_id == TEAM1_SHIRT_COLOUR
    assert bboxes[1].cluster_id == TEAM1_SHIRT_COLOUR
    assert bboxes[2].cluster_id == TEAM2_SHIRT_COLOUR
