from typing import Any
from logging import getLogger

from scorevision.utils.data_models import SVChallenge, SVRunOutput, SVEvaluation
from scorevision.vlm_pipeline.vlm_as_judge import (
    pairwise_judge_annotations_for_select_frames,
)
from scorevision.utils.settings import get_settings
from scorevision.vlm_pipeline.utils.data_models import (
    PseudoGroundTruth,
    MinerScore,
    AggregatedScore,
)
from scorevision.vlm_pipeline.utils.response_models import FrameAnnotation, BoundingBox
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.utils.smoothness import bbox_smoothness

logger = getLogger(__name__)


def parse_miner_prediction(miner_run: SVRunOutput) -> dict[int, FrameAnnotation]:
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
                bboxes.append(
                    BoundingBox(
                        bbox_2d=[
                            int(bbox["x1"]),
                            int(bbox["y1"]),
                            int(bbox["x2"]),
                            int(bbox["y2"]),
                        ],
                        label=ObjectOfInterest(str(bbox.get("cls", "player"))),
                        cluster_id=0,
                    )
                )
            except Exception as e:
                logger.error(e)
                continue
        miner_annotations[frame_number] = FrameAnnotation(
            bboxes=bboxes,
            category=Action.NONE,
            confidence=100,
            reason="",
        )
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
    miner_run: SVRunOutput, challenge: SVChallenge, miner_score: MinerScore
) -> SVEvaluation:
    """Final Score calculations considering VLM-as-Judge, Object Detection Smoothness across Video, Latency, etc..."""
    settings = get_settings()
    miner_annotations = parse_miner_prediction(miner_run=miner_run)

    vlm_as_judge_score = miner_score.score.total

    if not miner_run.success:
        logger.warning("Miner call did not successfully complete")
        smoothness_score = 0.0
    elif len(miner_annotations) != settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER:
        logger.warning(
            f"Miner did not predict expected number of frames ({len(miner_annotations)} != {settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER})"
        )
        smoothness_score = 0.0
    else:
        logger.info("Evaluating Smoothness of miner predictions")
        bboxes = [
            miner_annotations[frame_num].bboxes
            for frame_num in sorted(miner_annotations.keys())
        ]
        smoothness_score = bbox_smoothness(
            video_bboxes=bboxes,
            image_height=settings.SCOREVISION_IMAGE_HEIGHT,
            image_width=settings.SCOREVISION_IMAGE_WIDTH,
        )

    quality_score = (0.7 * vlm_as_judge_score + 0.3 * smoothness_score) / 2
    over = max(
        0.0,
        miner_run.latency_ms - settings.SCOREVISION_SCORE_SERVICE_LEVEL_OBJECTIVE_MS,
    )
    final_score = (quality_score**settings.SCOREVISION_SCORE_ALPHA) * (
        settings.SCOREVISION_SCORE_BASE
        ** (-settings.SCOREVISION_SCORE_MS_PENALTY * over)
    )
    logger.info(
        f"VLM-as-Judge = {vlm_as_judge_score}\nSmoothness = {smoothness_score}\nQuality = {quality_score}\nLatency = {miner_run.latency_ms}\nFinal Score = {final_score}"
    )
    details = {
        "judge_feedback": [
            feedback.model_dump(mode="json")
            for feedback in miner_score.vlm_as_judge_feedback
        ],
        "n_frames": len(miner_score.vlm_as_judge_feedback),
        "challenge_id": challenge.challenge_id,
        "prompt": challenge.prompt,
    }
    return SVEvaluation(
        acc_breakdown=miner_score.score.breakdown,
        acc=vlm_as_judge_score,
        smoothness=smoothness_score,
        latency_ms=miner_run.latency_ms,
        score=final_score,
        details=details,
    )
