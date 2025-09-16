from typing import Any
from logging import getLogger

from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
    CHALLENGE_ID_LOOKUP,
)
from scorevision.utils.data_models import (
    SVChallenge,
    SVRunOutput,
    SVEvaluation,
    TotalScore,
    SVPredictInput,
)

from scorevision.vlm_pipeline.non_vlm_scoring.keypoints import evaluate_keypoints
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_object_counts,
    compare_team_labels,
    compare_object_labels,
    compare_object_placement,
)
from scorevision.vlm_pipeline.vlm_as_judge import (
    pairwise_judge_annotations_for_select_frames,
)
from scorevision.utils.settings import get_settings
from scorevision.vlm_pipeline.utils.data_models import (
    PseudoGroundTruth,
    MinerScore,
    AggregatedScore,
)
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    BoundingBox,
    ShirtColor,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
    OBJECT_ID_LOOKUP,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import bbox_smoothness

logger = getLogger(__name__)


def parse_miner_prediction(miner_run: SVRunOutput) -> dict[int, dict]:
    predicted_frames = (
        (miner_run.predictions or {}).get("frames") if miner_run.predictions else None
    ) or []
    logger.info(f"Miner predicted {len(predicted_frames)} frames")

    miner_annotations = {}
    for predicted_frame in predicted_frames:
        bboxes = []
        frame_number = predicted_frame.get("frame_id", -1)
        for bbox in predicted_frame.get("boxes", []) or []:
            try:
                object_id = int(bbox.get("cls_id"))
                object_type = OBJECT_ID_LOOKUP.get(object_id)
                if object_type is None:
                    object_type = (
                        ObjectOfInterest.PLAYER
                    )  # NOTE: this assumes player is always a constant value in every challenge type
                    object_colour = ShirtColor.OTHER
                elif isinstance(object_type, str):
                    object_type = ObjectOfInterest.PLAYER
                    if "1" in object_type:
                        object_colour = TEAM1_SHIRT_COLOUR
                    else:
                        object_colour = TEAM2_SHIRT_COLOUR
                elif isinstance(object_type, ObjectOfInterest):
                    object_type = object_type
                    object_colour = ShirtColor.OTHER

                bboxes.append(
                    BoundingBox(
                        bbox_2d=[
                            int(bbox["x1"]),
                            int(bbox["y1"]),
                            int(bbox["x2"]),
                            int(bbox["y2"]),
                        ],
                        label=object_type,
                        cluster_id=object_colour,
                    )
                )
            except Exception as e:
                logger.error(e)
                continue
        miner_annotations[frame_number] = {
            "bboxes": bboxes,
            "action": predicted_frame.get("action", None),
            "keypoints": predicted_frame.get("keypoints", []),
        }
    return miner_annotations


async def evaluate_using_vlms(
    challenge: SVChallenge,
    miner_run: SVRunOutput,
    pseudo_gt_annotations: list[PseudoGroundTruth],
) -> MinerScore:

    settings = get_settings()

    miner_annotations = parse_miner_prediction(miner_run=miner_run)
    miner_id = miner_run.model or ""

    # VLM-as-Judge
    if not miner_run.success:
        logger.warning("Miner call did not successfully complete")
        miner_score = None
    elif len(miner_annotations) != settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER:
        logger.warning(
            f"Miner did not predict expected number of frames ({len(miner_annotations)} != {settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER})"
        )
        miner_score = None
    else:
        logger.info("Evaluating miner predictions with VLM-as-Judge")
        try:
            raise Exception("VLM-as-Judge is not scalable to be used at the moment")
            miner_score = await pairwise_judge_annotations_for_select_frames(
                miner_id=miner_id,
                video_url=challenge.challenge_id,
                challenge_data=pseudo_gt_annotations,
                miner_annotations=miner_annotations,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            miner_score = None

    if miner_score is None:
        logger.info("Evaluation scores to 0.0")
        miner_score = MinerScore(
            miner_id=miner_id,
            score=AggregatedScore(total=0.0, breakdown={}),
            video_url=challenge.challenge_id,
            frame_numbers=[
                pseudo_gt_annotation.frame_number
                for pseudo_gt_annotation in pseudo_gt_annotations
            ],
            vlm_as_judge_feedback=[],
            miner_annotations=miner_annotations,
        )
    return miner_score


def post_vlm_ranking(
    payload: SVPredictInput,
    miner_run: SVRunOutput,
    challenge: SVChallenge,
    miner_score: MinerScore,
    pseudo_gt_annotations: list[PseudoGroundTruth],
) -> SVEvaluation:
    """Final Score calculations considering VLM-as-Judge, Object Detection Smoothness across Video, Latency, etc..."""

    score_breakdown = TotalScore()

    settings = get_settings()
    miner_annotations = parse_miner_prediction(miner_run=miner_run)
    logger.info(payload.meta)
    challenge_id = int(payload.meta.get("task_id", -1))
    challenge_type = CHALLENGE_ID_LOOKUP.get(challenge_id)

    if (
        miner_run.success
        and len(miner_annotations) == settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER
        and challenge_type
    ):
        score_breakdown.keypoints.floor_markings_alignment = evaluate_keypoints(
            frames=payload.frames,
            miner_predictions=miner_annotations,
            challenge_type=challenge_type,
        )
        score_breakdown.objects.bbox_placement = compare_object_placement(
            pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
        )
        score_breakdown.objects.categorisation = compare_object_labels(
            pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
        )
        score_breakdown.objects.team = compare_team_labels(
            pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
        )
        score_breakdown.objects.enumeration = compare_object_counts(
            pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
        )
        score_breakdown.objects.tracking_stability = bbox_smoothness(
            video_bboxes=[
                miner_annotations[frame_num]["bboxes"]
                for frame_num in sorted(miner_annotations.keys())
            ],
            image_height=settings.SCOREVISION_IMAGE_HEIGHT,
            image_width=settings.SCOREVISION_IMAGE_WIDTH,
        )
        score_breakdown.latency.inference = 1 / 2 ** (miner_run.latency_ms / 1000)
        # TODO score action spotting: score_breakdown.action.categorisation =
    else:
        logger.info(
            f"Miner was either not successful ({miner_run.success}) or did not return predictions for all frames in the video ({len(miner_annotations)}) or challenge type was invalid ({challenge_id}: {challenge_type})"
        )

    details = {
        "breakdown": score_breakdown.to_dict(),
        # "judge_feedback": [
        #    feedback.model_dump(mode="json")
        #    for feedback in miner_score.vlm_as_judge_feedback
        # ],
        # "n_frames": len(miner_score.vlm_as_judge_feedback),
        "challenge_id": challenge.challenge_id,
        "prompt": challenge.prompt,
    }
    logger.info(details)
    return SVEvaluation(
        acc_breakdown=miner_score.score.breakdown,
        latency_ms=miner_run.latency_ms,
        acc=score_breakdown.average,
        score=score_breakdown.average,
        details=details,
    )
