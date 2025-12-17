from asyncio import gather
from logging import getLogger

from numpy import ndarray

from scorevision.utils.manifest import Element
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.polygons import (
    sam3_predictions_to_bounding_boxes,
)
from scorevision.vlm_pipeline.sam3.detect_objects import sam3_chute
from scorevision.vlm_pipeline.sam3.detect_team_colours import sam3_extract_shirt_colours
from scorevision.utils.async_clients import get_semaphore
from scorevision.vlm_pipeline.utils.response_models import FrameAnnotation, Action

logger = getLogger(__name__)


async def generate_annotations_for_select_frame_sam3(
    video_name: str,
    frame_number: int,
    frame: ndarray,
    flow_frame: ndarray,
    element: Element,
) -> PseudoGroundTruth | None:
    threshold = 0.5
    mosaic = 0
    object_team_name = "player"
    object_names = element.objects
    logger.info(
        f"Object names: {object_names}"
    )  # e.g. ["ball", "referee", "goalkeeper", "player"]
    if not object_names:
        raise ValueError("Element has no objects specified for detection!")

    semaphore = get_semaphore()
    async with semaphore:
        # STEP 1: Extract Shirt Colour for Teams
        try:
            jersey_colours = await sam3_extract_shirt_colours(
                image=frame,
                shirt_keyword=f"{object_team_name} jersey",
                threshold=threshold,
                mosaic=mosaic,
            )
            logger.info(f"Shirt colours detected: {jersey_colours}")
        except Exception as e:
            logger.error(f"SAM3 failed to from frame {frame_number}: {e}")
            return None

        other_object_names = [
            object_name
            for object_name in object_names
            if object_name != object_team_name
        ]
        team_labels = []
        if jersey_colours:
            for colour, _ in jersey_colours.most_common(2):
                team_labels.append(f"{object_team_name} in {colour}")
        else:
            team_labels.append(object_team_name)
        logger.info(f"Team labels: {team_labels}")

        # STEP 2: Segment Image
        try:
            other_object_names.extend(team_labels)
            segmentations = await sam3_chute(
                image=frame,
                object_names=other_object_names,
                threshold=threshold,
                mosaic=mosaic,
            )
            if not segmentations:
                raise ValueError("No objects returned")
        except Exception as e:
            logger.error(
                f"SAM3 failed to generate pseudo-GT annotations for frame {frame_number}: {e}"
            )
            return None

        # STEP 3: Convert Segments -> Bounding Boxes
        bboxes = sam3_predictions_to_bounding_boxes(
            results=segmentations,
            image=frame,
            team_labels=team_labels,
            original_team_label=object_team_name,
        )

        return PseudoGroundTruth(
            video_name=video_name,
            frame_number=frame_number,
            spatial_image=frame,
            temporal_image=flow_frame,
            annotation=FrameAnnotation(
                bboxes=bboxes, category=Action.NONE, confidence=100, reason="Sam3"
            ),
        )


async def generate_annotations_for_select_frames_sam3(
    video_name: str,
    frames: list[ndarray],
    flow_frames: list[ndarray],
    frame_numbers: list[int],
    element: Element,
) -> list[PseudoGroundTruth]:
    tasks = [
        generate_annotations_for_select_frame_sam3(
            video_name=video_name,
            frame_number=frame_number,
            frame=frame,
            flow_frame=flow_frame,
            element=element,
        )
        for frame_number, frame, flow_frame in zip(
            frame_numbers, frames, flow_frames, strict=True
        )
    ]
    results = await gather(*tasks, return_exceptions=True)
    annotations: list[PseudoGroundTruth] = []
    for result in results:
        if isinstance(result, PseudoGroundTruth):
            annotations.append(result)
        else:
            logger.error(result)
    return annotations
