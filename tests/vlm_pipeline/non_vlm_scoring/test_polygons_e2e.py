import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.client_exceptions import ClientConnectorDNSError

from scorevision.utils.settings import get_settings
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.image_annotation.single import annotate_image
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_false_positive,
    compare_polygon_map50,
    compare_polygon_placement,
    compare_polygon_precision,
    compare_polygon_recall,
)
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, ShirtColor
from scorevision.vlm_pipeline.vlm_annotator_sam3 import generate_annotations_for_select_frame_sam3
from scorevision.utils.manifest import ElementPrefix


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SAM3_E2E") != "1",
    reason="Set RUN_SAM3_E2E=1 to run the live SAM3 end-to-end polygon test.",
)


def _load_test_frame() -> np.ndarray:
    video_path = Path("tests/test_data/videos/example_football.mp4")
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read test frame from {video_path}")
    return frame


def _to_polygon_box(box: BoundingBox) -> BoundingBox:
    if box.polygon:
        return box
    x1, y1, x2, y2 = box.bbox_2d
    return BoundingBox(
        bbox_2d=box.bbox_2d,
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        label=box.label,
        score=box.score,
        cluster_id=box.cluster_id,
    )


def _make_random_polygon(rng: np.random.Generator, width: int, height: int) -> list[tuple[int, int]]:
    x1 = int(rng.integers(0, max(1, width // 2)))
    y1 = int(rng.integers(0, max(1, height // 2)))
    x2 = int(rng.integers(max(x1 + 1, width // 2), width))
    y2 = int(rng.integers(max(y1 + 1, height // 2), height))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


@pytest.mark.asyncio
async def test_sam3_polygon_end_to_end(tmp_path):
    settings = get_settings()
    frame = _load_test_frame()
    flow = np.zeros_like(frame)
    element = type(
        "E",
        (),
        {
            "category": ElementPrefix.OBJECT_DETECTION,
            "objects": ["player", "ball"],
        },
    )()

    try:
        pgt = await generate_annotations_for_select_frame_sam3(
            video_name="e2e",
            frame_number=1,
            frame=frame,
            flow_frame=flow,
            element=element,
        )
    except (ClientConnectorDNSError, OSError) as exc:
        pytest.skip(f"SAM3 endpoint unavailable: {exc}")
    assert pgt is not None
    assert pgt.annotation.bboxes

    gt_boxes = [_to_polygon_box(box) for box in pgt.annotation.bboxes]
    pseudo_gt = [pgt]

    null_predictions = {1: {"polygons": []}}
    pgt_predictions = {1: {"polygons": gt_boxes}}

    rng = np.random.default_rng(7)
    random_predictions = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=_make_random_polygon(rng, settings.SCOREVISION_IMAGE_WIDTH, settings.SCOREVISION_IMAGE_HEIGHT),
                    label=gt_boxes[0].label if gt_boxes else "player",
                    cluster_id=ShirtColor.OTHER,
                )
                for _ in range(max(1, len(gt_boxes)))
            ]
        }
    }

    null_scores = {
        "iou": compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
        "map50": compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
        "count": compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
        "precision": compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
        "recall": compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
        "false_positive": compare_polygon_false_positive(pseudo_gt=pseudo_gt, miner_predictions=null_predictions),
    }
    random_scores = {
        "iou": compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
        "map50": compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
        "count": compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
        "precision": compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
        "recall": compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
        "false_positive": compare_polygon_false_positive(pseudo_gt=pseudo_gt, miner_predictions=random_predictions),
    }
    pgt_scores = {
        "iou": compare_polygon_placement(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
        "map50": compare_polygon_map50(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
        "count": compare_polygon_counts(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
        "precision": compare_polygon_precision(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
        "recall": compare_polygon_recall(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
        "false_positive": compare_polygon_false_positive(pseudo_gt=pseudo_gt, miner_predictions=pgt_predictions),
    }

    assert pgt_scores["map50"] >= random_scores["map50"] >= null_scores["map50"]
    assert pgt_scores["precision"] >= random_scores["precision"] >= null_scores["precision"]
    assert pgt_scores["recall"] >= random_scores["recall"] >= null_scores["recall"]

    print("null_scores=", null_scores)
    print("random_scores=", random_scores)
    print("pgt_scores=", pgt_scores)

    null_preview = annotate_image(frame, FrameAnnotation(
        bboxes=[],
        category=Action.NONE,
        confidence=100,
        reason="null",
    ), "null")
    random_preview = annotate_image(frame, FrameAnnotation(
        bboxes=[_to_polygon_box(box) for box in random_predictions[1]["polygons"]],
        category=Action.NONE,
        confidence=100,
        reason="random",
    ), "random")
    pgt_preview = annotate_image(frame, pgt.annotation, "pgt")

    preview = np.concatenate([null_preview, random_preview, pgt_preview], axis=1)
    out_path = tmp_path / "sam3_polygon_e2e_preview.png"
    assert cv2.imwrite(str(out_path), preview)
    assert out_path.exists()
    print("preview_path=", out_path)
