from asyncio import gather
from logging import getLogger

from scorevision.vlm_pipeline.utils.prompts import (
    SYSTEM_PROMPT_VLM_AS_JUDGE,
    USER_PROMPT_VLM_AS_JUDGE,
)
from scorevision.utils.async_clients import get_semaphore
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    VLMJudgeFrameResults,
    Winner,
)
from scorevision.vlm_pipeline.utils.data_models import (
    Miner,
    MinerScore,
    AggregatedScore,
    PseudoGroundTruth,
)
from scorevision.vlm_pipeline.image_annotation.single import (
    annotate_image,
)
from scorevision.vlm_pipeline.image_annotation.pairwise import (
    display_iou,
)
from scorevision.vlm_pipeline.utils.llm_vlm import async_vlm_api, retry_api, VLMProvider
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action

logger = getLogger(__name__)


def quantify_judgement(winner: Winner) -> float:
    if winner == Winner.MINER:
        return 1.0
    elif winner == Winner.TIE:
        return 0.5
    return 0.0


def aggregate_results_over_frames(
    results: list[VLMJudgeFrameResults],
) -> AggregatedScore:
    detection_scores = []
    bbox_scores = []
    label_scores = []
    category_scores = []
    for result in results:
        detection_score = quantify_judgement(winner=result.detections.winner)
        logger.debug(f"Detection: {result.detections.winner} = {detection_score}")
        detection_scores.append(detection_score)

        bbox_score = quantify_judgement(winner=result.bboxes.winner)
        bbox_scores.append(bbox_score)
        logger.debug(f"BBoxes: {result.bboxes.winner} = {bbox_score}")

        labels_score = quantify_judgement(winner=result.labels.winner)
        label_scores.append(labels_score)
        logger.debug(f"Labels: {result.labels.winner} = {labels_score}")

        category_score = quantify_judgement(winner=result.category.winner)
        category_scores.append(category_score)
        logger.debug(f"Category: {result.category.winner} = {category_score}")

    logger.info(f"Detection: {detection_scores}")
    mean_detection_score = sum(detection_scores) / len(detection_scores)
    logger.info(f"BBox: {bbox_scores}")
    mean_bbox_score = sum(bbox_scores) / len(bbox_scores)
    logger.info(f"Label: {label_scores}")
    mean_label_score = sum(label_scores) / len(label_scores)
    logger.info(f"Category: {category_scores}")
    mean_category_score = sum(category_scores) / len(category_scores)

    weights = [0.3, 0.3, 0.25, 0.15]
    logger.info(
        f"Detection: {mean_detection_score}*{weights[0]} + BBox: {mean_bbox_score}*{weights[1]} + Label: {mean_label_score}*{weights[2]} + Category: {mean_category_score}*{weights[3]}"
    )
    scores = [
        mean_detection_score,
        mean_bbox_score,
        mean_label_score,
        mean_category_score,
    ]
    score = sum(w * s for w, s in zip(weights, scores))
    logger.info(f"Score: {score}")
    return AggregatedScore(
        total=score,
        breakdown=dict(
            zip(
                [
                    "Mean Detection Score",
                    "Mean BBox Score",
                    "Mean Label Score",
                    "Mean Category Score",
                ],
                scores,
                strict=True,
            )
        ),
    )


@retry_api
async def pairwise_judge_annotations_for_select_frame(
    miner_id: str,
    frame_data: PseudoGroundTruth,
    miner_annotation: dict | None,
    provider: VLMProvider = VLMProvider.PRIMARY,
) -> VLMJudgeFrameResults | None:

    if not miner_annotation:
        logger.error("No frame annotation provided - skipping VLM-as-judge")
        return

    semaphore = get_semaphore()
    async with semaphore:
        logger.info(f"Judging miner: {miner_id} on Frame: {frame_data.frame_number}...")
        h, w = frame_data.spatial_image.shape[:2]
        annotated_a = annotate_image(
            image=frame_data.spatial_image,
            annotations=frame_data.annotation,
            name=Winner.PSEUDO_GT.value,
        )
        annotated_b = annotate_image(
            image=frame_data.spatial_image,
            annotations=FrameAnnotation(
                bboxes=miner_annotation["bboxes"],
                category=Action(miner_annotation["action"]),
                confidence=100,
                reason="",
            ),
            name=Winner.MINER.value,
        )
        iou_image = display_iou(
            bboxes_a=frame_data.annotation.bboxes,
            bboxes_b=miner_annotation["bboxes"],
            image_height=h,
            image_width=w,
        )
        try:
            result_json = await async_vlm_api(
                images=[annotated_a, annotated_b, iou_image],
                system_prompt=SYSTEM_PROMPT_VLM_AS_JUDGE.format(
                    json_schema=VLMJudgeFrameResults.model_json_schema()
                ),
                user_prompt=USER_PROMPT_VLM_AS_JUDGE,
                provider=provider,
            )
            result = VLMJudgeFrameResults(**result_json)
        except Exception as e:
            logger.error(e)
            result = None
        return result


async def pairwise_judge_annotations_for_select_frames(
    miner_id: str,
    video_url: str,
    challenge_data: list[PseudoGroundTruth],
    miner_annotations: dict[int, dict],
) -> MinerScore:
    judge_results = []

    tasks = [
        pairwise_judge_annotations_for_select_frame(
            miner_id=miner_id,
            frame_data=frame_data,
            miner_annotation=miner_annotations.get(frame_data.frame_number),
        )
        for frame_data in challenge_data
    ]
    results = await gather(*tasks, return_exceptions=True)
    judge_results = []
    for result in results:
        if isinstance(result, VLMJudgeFrameResults):
            judge_results.append(result)
        else:
            logger.error(result)

    if not any(judge_results):
        raise ValueError(f"No Frames were successfully judged for miner: {miner_id}")

    return MinerScore(
        miner_id=miner_id,
        score=aggregate_results_over_frames(results=judge_results),
        video_url=video_url,
        frame_numbers=[frame_data.frame_number for frame_data in challenge_data],
        vlm_as_judge_feedback=judge_results,
        miner_annotations=[
            miner_annotations[frame_data.frame_number] for frame_data in challenge_data
        ],
    )
