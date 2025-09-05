from asyncio import Semaphore, gather
from logging import getLogger

from numpy import ndarray

from scorevision.vlm_pipeline.utils.prompts import (
    SYSTEM_PROMPT_VLM_ANNOTATOR,
    USER_PROMPT_VLM_ANNOTATOR,
)
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.llm_vlm import async_vlm_api, retry_api, VLMProvider
from scorevision.vlm_pipeline.utils.response_models import FrameAnnotation
from scorevision.utils.async_clients import get_semaphore

logger = getLogger(__name__)


@retry_api
async def generate_annotations_for_select_frame(
    video_name: str,
    frame_number: int,
    frame: ndarray,
    flow_frame: ndarray,
    provider: VLMProvider = VLMProvider.PRIMARY,
) -> PseudoGroundTruth | None:

    semaphore = get_semaphore()
    async with semaphore:
        try:
            annotation_json = await async_vlm_api(
                images=[frame, flow_frame],
                system_prompt=SYSTEM_PROMPT_VLM_ANNOTATOR.format(
                    json_schema=FrameAnnotation.model_json_schema()
                ),
                user_prompt=USER_PROMPT_VLM_ANNOTATOR,
                provider=provider,
            )
            annotation = FrameAnnotation(**annotation_json)
        except Exception as e:
            logger.error(
                f"VLM failed to generate pseudo-GT annotations for frame {frame_number}: {e}"
            )
            return None

        if not any(annotation.bboxes):
            logger.error("No annotations were generated for this frame")
            return

        return PseudoGroundTruth(
            video_name=video_name,
            frame_number=frame_number,
            spatial_image=frame,
            temporal_image=flow_frame,
            annotation=annotation,
        )


async def generate_annotations_for_select_frames(
    video_name: str,
    frames: list[ndarray],
    flow_frames: list[ndarray],
    frame_numbers: list[int],
) -> list[PseudoGroundTruth]:
    tasks = [
        generate_annotations_for_select_frame(
            video_name=video_name,
            frame_number=frame_number,
            frame=frame,
            flow_frame=flow_frame,
        )
        for frame_number, frame, flow_frame in zip(
            frame_numbers, frames, flow_frames, strict=True
        )
    ]
    results = await gather(*tasks, return_exceptions=True)
    annotations = []
    for result in results:
        if isinstance(result, PseudoGroundTruth):
            annotations.append(result)
        else:
            logger.error(result)
    return annotations
