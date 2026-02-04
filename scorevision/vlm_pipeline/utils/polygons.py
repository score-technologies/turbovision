from numpy import ndarray

from scorevision.vlm_pipeline.utils.response_models import BoundingBox
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    ShirtColor,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.vlm_pipeline.sam3.schemas import Sam3Result

TEAM_COLOURS = [TEAM1_SHIRT_COLOUR, TEAM2_SHIRT_COLOUR]


def bbox_from_polygon(
    polygon: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def sam3_predictions_to_bounding_boxes(
    results: list[Sam3Result],
    image: ndarray,
    team_labels: list[str],
    original_team_label: str,
) -> list[BoundingBox]:
    bboxes = []
    for result in results:
        object_label_raw = result.echo.text
        if object_label_raw in team_labels:
            object_label = original_team_label
            team_index = team_labels.index(object_label_raw)
            colour = TEAM_COLOURS[team_index]
        else:
            object_label = object_label_raw
            colour = ShirtColor.OTHER
        for prediction in result.predictions:
            for polygon in prediction.masks:

                bbox = BoundingBox(
                    bbox_2d=bbox_from_polygon(polygon=polygon),
                    label=object_label,
                    cluster_id=colour,
                )
                bboxes.append(bbox)
    return bboxes
