if __name__ == "__main__":
    from logging import basicConfig, INFO

    from asyncio import run
    from cv2 import imread, imshow, waitKey

    from scorevision.vlm_pipeline.utils.polygons import (
        sam3_predictions_to_bounding_boxes,
    )
    from scorevision.vlm_pipeline.image_annotation.single import annotate_bbox
    from scorevision.vlm_pipeline.sam3.detect_objects import sam3_chute

    basicConfig(level=INFO)
    image = imread("vlm_pipeline/sam3/apples.png")
    results = run(
        sam3_chute(
            image=image, object_names=["apple"], threshold=0.5, mosaic=1
        )
    )
    bboxes = sam3_predictions_to_bounding_boxes(results=results, image=image)
    for bbox in bboxes:
        annotate_bbox(frame=image, bbox=bbox)

    imshow("Annotated Image", image)
    waitKey(0)
