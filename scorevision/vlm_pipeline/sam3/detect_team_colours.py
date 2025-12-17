from numpy import ndarray, zeros, array, mean
from collections import Counter
from logging import getLogger

from scipy.spatial import KDTree
from cv2 import fillPoly

from scorevision.vlm_pipeline.sam3.detect_objects import sam3_chute
from scorevision.vlm_pipeline.sam3.schemas import Sam3Result

logger = getLogger(__name__)

def name_of_colour(bgr_pixel:tuple[int,int,int]) -> str:
    COLOR_MAP = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
    }
    COLOR_NAMES = list(COLOR_MAP.keys())
    TREE = KDTree(list(COLOR_MAP.values()))

    _, index = TREE.query(bgr_pixel)
    return COLOR_NAMES[index]

def get_roi(image:ndarray, polygon:list[tuple[int,int]]) -> ndarray:
    mask = zeros(image.shape[:2], dtype="uint8")
    poly_pts = array(polygon, dtype="int32")
    fillPoly(mask, [poly_pts], 255)
    return image[mask == 255]

async def sam3_extract_shirt_colours(image:ndarray) -> Counter:
    segmentation_results = await sam3_chute(
        image=image, object_names=["player jersey"], threshold=0.5, mosaic=0
    )
    detected_colours = []
    if segmentation_results: 
        for segmentation_result in segmentation_results:
            for prediction in segmentation_result.predictions:
                for polygon in prediction.masks:
                    roi = get_roi(image,polygon)
                    if not len(roi):
                        continue
                    avg_rgb = mean(roi, axis=0)
                    colour_name = name_of_colour(tuple(avg_rgb))
                    detected_colours.append(colour_name)
    return Counter(detected_colours)



async def detect_team_players(image:ndarray, object_names:list[str]) -> list[Sam3Result]:
    jersey_colours = await sam3_extract_shirt_colours(image)
    logger.info(jersey_colours)

    player_labels = []
    if jersey_colours:
        for colour, _ in jersey_colours.most_common(2):
            player_labels.append(f"player in {colour}")
    else:
        player_labels.append("player")

    logger.info(player_labels)
    object_names.extend(player_labels)
    return await sam3_chute(
        image=image, object_names=object_names, threshold=0.5, mosaic=0
    )

