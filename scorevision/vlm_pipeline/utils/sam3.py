from logging import getLogger

from numpy import ndarray
from pydantic import BaseModel

from scorevision.utils.image_processing import image_to_b64string
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.chutes_helpers import warmup_chute

logger = getLogger(__name__)


class Polygons(BaseModel):
    masks: list[list[tuple[int, int]]]
    confidence: float


class ObjectName(BaseModel):
    text: str
    num_boxes: int


class Sam3Result(BaseModel):
    prompt_index: int
    echo: ObjectName
    predictions: list[Polygons]


async def sam3_chute(image: ndarray, object_names: list[str], threshold: float) -> dict:
    settings = get_settings()
    endpoint = settings.CHUTES_SAM3_ENDPOINT
    headers = {
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
        "Content-Type": "application/json",
    }
    payload = {
        "image": {"type": "base64", "value": image_to_b64string(image=image)},
        "prompts": [
            {"type": "text", "text": object_name} for object_name in object_names
        ],
        "output_prob_thresh": threshold,
    }
    for attempt in range(settings.SCOREVISION_API_N_RETRIES):
        logger.info(
            f"Calling SAM3 via Chutes attempt {attempt+1}/{settings.SCOREVISION_API_N_RETRIES}..."
        )
        session = await get_async_client()
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 200:
                    response_json = await response.json()
                    logger.info(response_json)
                    message_content = response_json.get("prompt_results", [])
                    return message_content
                elif response.status == 503:
                    await warmup_chute(chute_id=settings.CHUTES_SAM3_ID)
                    raise Exception(
                        f"API request failed with status {response.status} (chute cold): {await response.text()}. Chute now warmed up..."
                    )
                raise Exception(
                    f"API request failed with status {response.status}: {await response.text()}"
                )
        except Exception as e:
            wait_time = min(attempt, settings.SCOREVISION_API_RETRY_DELAY_S)
            logger.info(f"API request failed: {e}. Retrying in {wait_time} s...")


if __name__ == "__main__":
    from logging import basicConfig, INFO

    from asyncio import run
    from cv2 import imread, imshow, waitKey

    from scorevision.vlm_pipeline.domain_specific_schemas.football import Person
    from scorevision.vlm_pipeline.utils.polygons import (
        sam3_predictions_to_bounding_boxes,
    )
    from scorevision.vlm_pipeline.image_annotation.single import annotate_bbox

    basicConfig(level=INFO)
    image = imread("test.jpg")  # "test-bb.png")
    result = run(
        sam3_chute(
            image=image, object_names=[person.value for person in Person], threshold=0.5
        )
    )
    results = [Sam3Result(**r) for r in result]
    bboxes = sam3_predictions_to_bounding_boxes(results=results, image=image)
    for bbox in bboxes:
        annotate_bbox(frame=image, bbox=bbox)

    imshow("Annotated Image", image)
    waitKey(0)
