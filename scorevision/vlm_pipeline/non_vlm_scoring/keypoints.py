from logging import getLogger
from typing import Any
import numpy as np
import cv2

from numpy import array, uint8, float32, ndarray
from cv2 import (
    bitwise_and,
    bitwise_not,
    bitwise_or,
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
    dilate,
)

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth

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


def has_a_wide_line(mask: ndarray, max_aspect_ratio: float = 1.0) -> bool:
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
    )[None, :, :]

    warped_corners = perspectiveTransform(src_corners, homography_matrix)[0]

    if is_bowtie(warped_corners):
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
        if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:  # ignore default / missing points
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
    if cv2.countNonZero(mask_ground_binary) == 0:
        raise InvalidMask("No projected ground (empty mask)")
    pts = cv2.findNonZero(mask_ground_binary)
    x, y, w, h = cv2.boundingRect(pts)
    is_rect = cv2.countNonZero(mask_ground_binary) == (w * h)

    if is_rect:
        raise InvalidMask(
            f"Projected ground should not be rectangular"
        )

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
    dilate_kernel_size: int = 3,  # thicken the edges
    dilate_iterations: int = 3,
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

    if dilate_kernel_size > 1:
        dilate_kernel = getStructuringElement(
            MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
        )
        image_edges_on_ground = dilate(
            image_edges_on_ground, dilate_kernel, iterations=dilate_iterations
        )
    return (image_edges_on_ground > 0).astype(uint8)

blacklists = [
    [23, 24, 27, 28],
    [7, 8, 3, 4],
    [2, 10, 1, 14],
    [18, 26, 14, 25],
    [5, 13, 6, 17],
    [21, 29, 17, 30],
    [10, 11, 2, 3],
    [10, 11, 2, 7],
    [12, 13, 4, 5],
    [12, 13, 5, 8],
    [18, 19, 26, 27],
    [18, 19, 26, 23],
    [20, 21, 24, 29],
    [20, 21, 28, 29],
    [8, 4, 5, 13],
    [3, 7, 2, 10],
    [23, 27, 18, 26],
    [24, 28, 21, 29]
]

def near_edges(x, y, W, H, t=50):
    edges = set()
    if x <= t:
        edges.add("left")
    if x >= W - t:
        edges.add("right")
    if y <= t:
        edges.add("top")
    if y >= H - t:
        edges.add("bottom")
    return edges

def both_points_same_direction(A, B, W, H, t=100):
    edges_A = near_edges(A[0], A[1], W, H, t)
    edges_B = near_edges(B[0], B[1], W, H, t)

    if not edges_A or not edges_B:
        return False

    return not edges_A.isdisjoint(edges_B)

def evaluate_keypoints_for_frame(
    template_keypoints: list[tuple[int, int]],
    frame_keypoints: list[tuple[int, int]],
    frame: ndarray,
    floor_markings_template: ndarray,
) -> float:
    try:
        frame_height, frame_width = frame.shape[:2]

        non_idxs = []
        frame_keypoints = [
            (0, 0) if (x, y) != (0, 0) and (x < 0 or y < 0 or x >= frame_width or y >= frame_height) else (x, y)
            for (x, y) in frame_keypoints
        ]

        for idx, kpts in enumerate(frame_keypoints):
            if kpts[0] != 0 or kpts[1] != 0:
                non_idxs.append(idx + 1)

        for blacklist in blacklists:
            is_included = set(non_idxs).issubset(blacklist)
            if is_included:
                if both_points_same_direction(frame_keypoints[blacklist[0] - 1], frame_keypoints[blacklist[1] - 1], frame_width, frame_height):
                    logger.info(f"Suspect keypoints!")
                    return 0
        
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

        pixels_overlapping_result = bitwise_and(
            mask_lines_expected, mask_lines_predicted
        )

        ys, xs = np.where(mask_lines_expected == 1)

        if len(xs) == 0:
            bbox = None
        else:
            min_x = xs.min()
            max_x = xs.max()
            min_y = ys.min()
            max_y = ys.max()

            bbox = (min_x, min_y, max_x, max_y)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox is not None else 1
        frame_area = frame.shape[0] * frame.shape[1]

        if (bbox_area / frame_area) < 0.2:
            return 0.0

        valid_keypoints = [
            (x, y) for x, y in frame_keypoints
            if not (x == 0 and y == 0)
        ]
        if not valid_keypoints:
            return 0.0
        xs, ys = zip(* valid_keypoints)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if max_x < 0 or max_y < 0 or min_x >= frame_width or min_y >= frame_height:
            logger.info("All keypoints are outside the frame")
            return 0.0

        if (max_x - min_x) > 2 * frame_width or (max_y - min_y) > 2 * frame_height:
            logger.info("Keypoints spread too wide")
            return 0.0

        inv_expected = bitwise_not(mask_lines_expected)
        pixels_rest = bitwise_and(inv_expected, mask_lines_predicted).sum()

        total_pixels = bitwise_or(mask_lines_expected, mask_lines_predicted).sum()

        if total_pixels == 0:
            return 0.0

        if (pixels_rest / (total_pixels)) > 0.9:
            logger.info('threshold exceeded')
            return 0.0

        pixels_overlapping = pixels_overlapping_result.sum()
        pixels_on_lines = mask_lines_expected.sum()
        score = pixels_overlapping / (pixels_on_lines + 1e-8)
        return score
    except Exception as e:
        logger.error(e)
    return 0.0


def evaluate_keypoints(
    miner_predictions: dict[int, dict],
    frames: Any,
    challenge_type: SVChallenge,
) -> float:
    # TODO: use challenge_type to switch the template and keypoints
    template_image = challenge_template()
    template_keypoints = KEYPOINTS
    # frame_lookup = {frame.frame_id: frame for frame in frames}
    frame_scores = []
    for frame_number, annotations_miner in miner_predictions.items():
        miner_keypoints = annotations_miner["keypoints"]
        frame_image = None
        if frames is not None:
            if hasattr(frames, "get_frame"):
                try:
                    frame_image = frames.get_frame(frame_number)
                except Exception as e:
                    logger.error(e)
                    frame_image = None
            else:
                try:
                    frame_image = frames.get(frame_number)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.error(e)
                    frame_image = None
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
                frame=frame_image,  # array(frame_image.image),
                floor_markings_template=template_image.copy(),
            )
        logger.info(f"[evaluate_keypoints] Frame {frame_number}: {frame_score}")
        frame_scores.append(frame_score)
    return sum(frame_scores) / len(frame_scores)
