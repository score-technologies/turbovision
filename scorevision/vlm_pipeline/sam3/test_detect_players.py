
if __name__ == "__main__":
    from logging import basicConfig, INFO

    from asyncio import run
    from cv2 import imread, imwrite

    from scorevision.vlm_pipeline.utils.polygons import (
        sam3_predictions_to_bounding_boxes,
    )
    from scorevision.vlm_pipeline.image_annotation.single import annotate_bbox
    from scorevision.vlm_pipeline.sam3.detect_team_colours import detect_team_players

    basicConfig(level=INFO)
    image = imread("vlm_pipeline/sam3/football.jpg")
    results = run(detect_team_players(image=image, object_names = ["ball", "referee", "goalkeeper"]))
    bboxes = sam3_predictions_to_bounding_boxes(results=results, image=image)
    if bboxes:
        for bbox in bboxes:
            annotate_bbox(frame=image, bbox=bbox)

    imwrite("vlm_pipeline/sam3/football_annotated.png", image)
