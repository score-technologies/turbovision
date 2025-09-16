from logging import getLogger

from numpy import array, uint8, float32, ndarray, logical_and
from cv2 import (
    bitwise_and,
    findHomography,
    warpPerspective,
    cvtColor,
    COLOR_BGR2GRAY,
    threshold,
    THRESH_BINARY,
    getStructuringElement,
    MORPH_RECT,
    MORPH_TOPHAT,
    GaussianBlur,
    morphologyEx,
    Canny,
    connectedComponents,
    perspectiveTransform,
    RETR_EXTERNAL,
    CHAIN_APPROX_SIMPLE,
    findContours,
    boundingRect,
)

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth

# from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import intersection_over_union
from scorevision.chute_template.schemas import SVFrame
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    football_pitch as challenge_template,
    FOOTBALL_KEYPOINTS as KEYPOINTS,
    INDEX_KEYPOINT_CORNER_BOTTOM_LEFT,
    INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT,
    INDEX_KEYPOINT_CORNER_TOP_LEFT,
    INDEX_KEYPOINT_CORNER_TOP_RIGHT,
)
from scorevision.utils.data_models import SVChallenge

logger = getLogger(__name__)


class InvalidMask(Exception):
    pass


def has_a_wide_line(mask: ndarray, max_aspect_ratio: float = 0.5) -> bool:
    contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio >= max_aspect_ratio:
            return True
    return False


def is_bowtie(points: ndarray) -> bool:
    def segments_intersect(p1: int, p2: int, q1: int, q2: int) -> bool:
        def ccw(a: int, b: int, c: int):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (
            ccw(p1, p2, q1) != ccw(p1, p2, q2)
        )

    pts = points.reshape(-1, 2)
    edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
    return segments_intersect(*edges[0], *edges[2]) or segments_intersect(
        *edges[1], *edges[3]
    )


def validate_mask_lines(mask: ndarray) -> None:
    if mask.sum() == 0:
        raise InvalidMask("No projected lines")
    if mask.sum() == mask.size:
        raise InvalidMask("Projected lines cover the entire image surface")
    if has_a_wide_line(mask=mask):
        raise InvalidMask("A projected line is too wide")


def validate_mask_ground(mask: ndarray) -> None:
    num_labels, _ = connectedComponents(mask)
    num_distinct_regions = num_labels - 1
    if num_distinct_regions > 1:
        raise InvalidMask(
            f"Projected ground should be a single object, detected {num_distinct_regions}"
        )
    area_covered = mask.sum() / mask.size
    if area_covered >= 0.9:
        raise InvalidMask(
            f"Projected ground covers more than {area_covered:.2f}% of the image surface which is unrealistic"
        )


def validate_projected_corners(
    source_keypoints: list[tuple[int, int]], homography_matrix: ndarray
) -> None:
    src_corners = array(
        [
            source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_LEFT],
            source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT],
            source_keypoints[INDEX_KEYPOINT_CORNER_TOP_RIGHT],
            source_keypoints[INDEX_KEYPOINT_CORNER_TOP_LEFT],
        ],
        dtype="float32",
    )
    src_corners_reshaped = src_corners[None, :, :]
    warped_corners = perspectiveTransform(src_corners_reshaped, homography_matrix)[0]
    cnt = warped_corners.reshape((-1, 1, 2))
    if is_bowtie:
        raise InvalidMask("Projection twisted!")


def project_image_using_keypoints(
    image: ndarray,
    source_keypoints: list[tuple[int, int]],
    destination_keypoints: list[tuple[int, int]],
    destination_width: int,
    destination_height: int,
    inverse: bool = False,
) -> ndarray:
    filtered_src = []
    filtered_dst = []

    for src_pt, dst_pt in zip(source_keypoints, destination_keypoints, strict=True):
        if dst_pt == (0.0, 0.0):  # ignore default / missing points
            continue
        filtered_src.append(src_pt)
        filtered_dst.append(dst_pt)

    if len(filtered_src) < 4:
        raise ValueError("At least 4 valid keypoints are required for homography.")

    source_points = array(filtered_src, dtype=float32)
    destination_points = array(filtered_dst, dtype=float32)

    if inverse:
        H_inv, _ = findHomography(destination_points, source_points)
        return warpPerspective(image, H_inv, (destination_width, destination_height))
    H, _ = findHomography(source_points, destination_points)
    projected_image = warpPerspective(image, H, (destination_width, destination_height))
    validate_projected_corners(source_keypoints=source_keypoints, homography_matrix=H)
    return projected_image


def extract_masks_for_ground_and_lines(
    image: ndarray,
) -> tuple[ndarray, ndarray]:
    """assumes template coloured s.t. ground = gray, lines = white, background = black"""

    gray = cvtColor(image, COLOR_BGR2GRAY)
    _, mask_ground = threshold(gray, 10, 255, THRESH_BINARY)
    _, mask_lines = threshold(gray, 200, 255, THRESH_BINARY)
    mask_ground_binary = (mask_ground > 0).astype(uint8)
    mask_lines_binary = (mask_lines > 0).astype(uint8)

    validate_mask_ground(mask=mask_ground_binary)
    validate_mask_lines(mask=mask_lines_binary)
    return mask_ground_binary, mask_lines_binary


def extract_mask_of_ground_lines_in_image(
    image: ndarray,
    ground_mask: ndarray,
    blur_ksize: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    use_tophat: bool = True,
) -> ndarray:
    h, w = image.shape[:2]
    gray = cvtColor(image, COLOR_BGR2GRAY)
    if use_tophat:
        kernel = getStructuringElement(MORPH_RECT, (31, 31))
        gray = morphologyEx(gray, MORPH_TOPHAT, kernel)

    if blur_ksize and blur_ksize % 2 == 1:
        gray = GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    image_edges = Canny(gray, canny_low, canny_high)
    image_edges_on_ground = bitwise_and(image_edges, image_edges, mask=ground_mask)
    return (image_edges_on_ground > 0).astype(uint8)


def evaluate_keypoints_for_frame(
    template_keypoints: list[tuple[int, int]],
    frame_keypoints: list[tuple[int, int]],
    frame: ndarray,
    floor_markings_template: ndarray,
) -> float:
    try:
        warped_template = project_image_using_keypoints(
            image=floor_markings_template,
            source_keypoints=template_keypoints,
            destination_keypoints=frame_keypoints,
            destination_width=frame.shape[1],
            destination_height=frame.shape[0],
        )
        mask_ground, mask_lines_expected = extract_masks_for_ground_and_lines(
            image=warped_template
        )
        mask_lines_predicted = extract_mask_of_ground_lines_in_image(
            image=frame, ground_mask=mask_ground
        )

        # score = intersection_over_union(
        #    mask1=mask_lines_expected, mask2=mask_lines_predicted
        # )
        pixels_overlapping = logical_and(
            mask_lines_expected, mask_lines_predicted
        ).sum()
        pixels_on_edges = mask_lines_predicted.sum()
        score = pixels_overlapping / (pixels_on_edges + 1e-8)
        return score
    except Exception as e:
        logger.error(e)
    return 0.0


def evaluate_keypoints(
    miner_predictions: dict[int, dict],
    frames: list[SVFrame],
    challenge_type: SVChallenge,
) -> list[float]:
    # TODO: use challenge_type to switch the template and keypoints
    template_image = challenge_template()
    template_keypoints = KEYPOINTS
    frame_lookup = {frame.frame_id: frame for frame in frames}
    frame_scores = []
    for frame_number, annotations_miner in miner_predictions.items():
        miner_keypoints = annotations_miner["keypoints"]
        frame_image = frame_lookup.get(frame_number)
        if (
            annotations_miner is None
            or frame_image is None
            or len(miner_keypoints) != len(KEYPOINTS)
        ):
            frame_score = 0.0
        else:
            frame_score = evaluate_keypoints_for_frame(
                template_keypoints=template_keypoints,
                frame_keypoints=miner_keypoints,
                frame=array(frame_image.image),
                floor_markings_template=template_image,
            )
        logger.info(f"[evaluate_keypoints] Frame {frame_number}: {frame_score}")
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)
