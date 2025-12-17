from scorevision.vlm_pipeline.sam3.schemas import Sam3Result
from scorevision.vlm_pipeline.sam3.detect_objects import sam3_chute
from scorevision.vlm_pipeline.sam3.detect_team_colours import sam3_extract_shirt_colours


async def detect_players() -> list[Sam3Result]:
    object_names = ["ball", "referee", "goalkeeper"]
    return await sam3_chute(
        image=image, object_names=object_names, threshold=0.5, mosaic=0
    )

if __name__ == "__main__":
    from logging import basicConfig, INFO

    from asyncio import run
    from cv2 import imread, imshow, waitKey

    from scorevision.vlm_pipeline.utils.polygons import (
        sam3_predictions_to_bounding_boxes,
    )
    from scorevision.vlm_pipeline.image_annotation.single import annotate_bbox

    basicConfig(level=INFO)
    image = imread("vlm_pipeline/sam3/apples.png")
    results = run(detect_players(image=image))
    bboxes = sam3_predictions_to_bounding_boxes(results=results, image=image)
    for bbox in bboxes:
        annotate_bbox(frame=image, bbox=bbox)

    imshow("Annotated Image", image)
    waitKey(0)
