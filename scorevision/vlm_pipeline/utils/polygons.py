from numpy import zeros, uint8, int32, array, mean, ndarray, histogram, argmax
from numpy.linalg import norm
from cv2 import fillPoly, cvtColor, COLOR_BGR2HSV

from scorevision.vlm_pipeline.utils.response_models import BoundingBox
from scorevision.vlm_pipeline.domain_specific_schemas.football import Person, ShirtColor
from scorevision.vlm_pipeline.utils.sam3 import Sam3Result
from scorevision.vlm_pipeline.image_annotation.single import COLOURS


def colour_from_polygon(image: ndarray, polygon: list[tuple[int, int]]) -> ShirtColor:
    polygon_mask = zeros(image.shape[:2], dtype=uint8)
    pts = array(polygon, dtype=int32)
    fillPoly(polygon_mask, [pts], 255)
    pixels = image[polygon_mask == 255]
    if len(pixels) == 0:
        return ShirtColor.OTHER

    hsv_pixels = cvtColor(pixels.reshape(-1, 1, 3), COLOR_BGR2HSV).reshape(-1, 3)
    H, S, V = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]

    valid = S > 30
    if valid.sum() < 50:
        if V.mean() > 200:
            return ShirtColor.WHITE
        if V.mean() < 60:
            return ShirtColor.BLACK
        if 150 < V.mean() < 220:
            return ShirtColor.SILVER
        return ShirtColor.GREY

    H_valid = H[valid]

    hist, bin_edges = histogram(H_valid, bins=36, range=(0, 180))
    top_bin = argmax(hist)
    dominant_hue = (bin_edges[top_bin] + bin_edges[top_bin + 1]) / 2

    mean_saturation = S.mean()
    mean_value = V.mean()

    if mean_value > 200 and mean_saturation < 30:
        return ShirtColor.WHITE
    if mean_value < 60:
        return ShirtColor.BLACK
    if mean_saturation < 25:
        return ShirtColor.GREY

    if (0 <= dominant_hue < 10 or 170 <= dominant_hue <= 180) and mean_value < 120:
        return ShirtColor.MAROON
    if 10 <= dominant_hue < 35 and 60 < mean_value < 160:
        return ShirtColor.BROWN
    if 20 <= dominant_hue < 35 and mean_value > 180 and mean_saturation > 80:
        return ShirtColor.GOLD
    if 0 <= dominant_hue < 10 or 170 <= dominant_hue <= 180:
        return ShirtColor.RED
    if 10 <= dominant_hue < 20:
        return ShirtColor.ORANGE
    if 20 <= dominant_hue < 35:
        return ShirtColor.YELLOW
    if 35 <= dominant_hue < 85:
        return ShirtColor.GREEN
    if 85 <= dominant_hue < 110:
        return ShirtColor.TURQUOISE
    if 110 <= dominant_hue < 130:
        return ShirtColor.BLUE
    if 130 <= dominant_hue < 155:
        return ShirtColor.PURPLE
    if 155 <= dominant_hue < 170:
        return ShirtColor.PINK
    return ShirtColor.OTHER


def bbox_from_polygon(
    polygon: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def sam3_predictions_to_bounding_boxes(
    results: list[Sam3Result], image: ndarray
) -> list[BoundingBox]:
    bboxes = []
    for result in results:
        object_label = Person(result.echo.text)
        for prediction in result.predictions:
            for polygon in prediction.masks:
                colour = colour_from_polygon(polygon=polygon, image=image)
                bbox = BoundingBox(
                    bbox_2d=bbox_from_polygon(polygon=polygon),
                    label=object_label,
                    cluster_id=colour,
                )
                bboxes.append(bbox)
    return bboxes
