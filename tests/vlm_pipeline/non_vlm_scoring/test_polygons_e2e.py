from pathlib import Path

import cv2
import numpy as np
import pytest

from scorevision.utils.settings import get_settings
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.image_annotation.single import annotate_image
from scorevision.vlm_pipeline.sam3.schemas import ObjectName, Polygons, Sam3Result
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_false_positive,
    compare_polygon_map50,
    compare_polygon_placement,
    compare_polygon_precision,
    compare_polygon_recall,
)
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_false_positive,
    compare_map50,
    compare_object_counts,
    compare_object_placement,
    compare_precision,
    compare_recall,
)
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, ShirtColor
import scorevision.vlm_pipeline.vlm_annotator_sam3 as sam3_annotator
from scorevision.vlm_pipeline.vlm_annotator_sam3 import generate_annotations_for_select_frame_sam3
from scorevision.utils.manifest import ElementPrefix


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


def _bbox_only_box(box: BoundingBox) -> BoundingBox:
    return BoundingBox(
        bbox_2d=box.bbox_2d,
        label=box.label,
        score=box.score,
        cluster_id=box.cluster_id,
    )


def _make_random_polygon(rng: np.random.Generator, width: int, height: int) -> list[tuple[int, int]]:
    x1 = int(rng.integers(0, max(1, width // 2)))
    y1 = int(rng.integers(0, max(1, height // 2)))
    x2 = int(rng.integers(max(x1 + 1, width // 2), width))
    y2 = int(rng.integers(max(y1 + 1, height // 2), height))
    x_mid = min(x2 - 1, max(x1 + 1, (x1 + x2) // 2 + int(rng.integers(-8, 9))))
    y_mid = min(y2 - 1, max(y1 + 1, (y1 + y2) // 2 + int(rng.integers(-8, 9))))
    return [(x1, y1), (x2, y1), (x_mid, y_mid), (x1, y2)]


def _overlap_polygon_from_box(box: BoundingBox, shrink: int = 8) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = box.bbox_2d
    x1 = min(x2 - 2, x1 + shrink)
    y1 = min(y2 - 2, y1 + shrink)
    x2 = max(x1 + 2, x2 - shrink)
    y2 = max(y1 + 2, y2 - shrink)
    x_mid = max(x1 + 1, x2 - 3)
    y_mid = min(y2 - 1, y1 + max(2, (y2 - y1) // 2))
    return [(x1, y1), (x2, y1), (x_mid, y_mid), (x1, y2)]


def _poly_mask(points: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _mock_sam3_results() -> list[Sam3Result]:
    return [
        Sam3Result(
            prompt_index=0,
            echo=ObjectName(text="player", num_boxes=1),
            predictions=[
                Polygons(
                    confidence=0.99,
                    masks=[
                        [(10, 10), (72, 18), (64, 124), (16, 116)],
                    ],
                )
            ],
        ),
        Sam3Result(
            prompt_index=1,
            echo=ObjectName(text="ball", num_boxes=1),
            predictions=[
                Polygons(
                    confidence=0.95,
                    masks=[
                        [(150, 80), (178, 72), (185, 96), (160, 112)],
                    ],
                )
            ],
        ),
    ]


def _score_lines(scores: dict[str, float]) -> list[str]:
    return [
        f"iou={scores['iou']:.2f}",
        f"map50={scores['map50']:.2f}",
        f"count={scores['count']:.2f}",
        f"precision={scores['precision']:.2f}",
        f"recall={scores['recall']:.2f}",
        f"fp={scores['false_positive']:.2f}",
    ]


def _build_mock_pgt(frame: np.ndarray):
    flow = np.zeros_like(frame)
    element = type(
        "E",
        (),
        {
            "category": ElementPrefix.OBJECT_DETECTION,
            "objects": ["player", "ball"],
        },
    )()
    return flow, element


async def _build_sam3_pgt(frame: np.ndarray, monkeypatch) -> tuple[object, list[BoundingBox]]:
    async def fake_detect_objects_sam3(*args, **kwargs):
        return sam3_annotator.sam3_predictions_to_bounding_boxes(
            results=_mock_sam3_results(),
            image=frame,
            team_labels=[],
            original_team_label="",
        )

    monkeypatch.setattr(sam3_annotator, "detect_objects_sam3", fake_detect_objects_sam3)
    flow, element = _build_mock_pgt(frame)
    pgt = await generate_annotations_for_select_frame_sam3(
        video_name="e2e",
        frame_number=1,
        frame=frame,
        flow_frame=flow,
        element=element,
    )
    assert pgt is not None
    assert pgt.annotation.bboxes
    return pgt, [_to_polygon_box(box) for box in pgt.annotation.bboxes]


@pytest.mark.asyncio
async def test_sam3_polygon_end_to_end(tmp_path, monkeypatch):
    settings = get_settings()
    frame = _load_test_frame()
    pgt, gt_boxes = await _build_sam3_pgt(frame, monkeypatch)
    pseudo_gt = [pgt]

    null_predictions = {1: {"polygons": []}}
    pgt_predictions = {1: {"polygons": gt_boxes}}

    rng = np.random.default_rng(7)
    random_count = max(10, len(gt_boxes) * 8)
    random_predictions = {
        1: {
            "polygons": [
                BoundingBox(
                    bbox_2d=gt_boxes[0].bbox_2d,
                    polygon=_overlap_polygon_from_box(gt_boxes[0], shrink=8),
                    label=gt_boxes[0].label if gt_boxes else "player",
                    cluster_id=ShirtColor.OTHER,
                )
            ]
            + [
                BoundingBox(
                    bbox_2d=(0, 0, 10, 10),
                    polygon=_make_random_polygon(rng, settings.SCOREVISION_IMAGE_WIDTH, settings.SCOREVISION_IMAGE_HEIGHT),
                    label=gt_boxes[0].label if gt_boxes else "player",
                    cluster_id=ShirtColor.OTHER,
                )
                for _ in range(random_count - 1)
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
    overlap_scores = random_scores

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

    overlap_panel = frame.copy()
    overlap_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for pred_box in random_predictions[1]["polygons"]:
        pred_mask = _poly_mask(pred_box.polygon or [], frame.shape[:2])
        for gt_box in gt_boxes:
            gt_mask = _poly_mask(gt_box.polygon or [], frame.shape[:2])
            overlap_mask = np.maximum(overlap_mask, np.logical_and(pred_mask, gt_mask).astype(np.uint8))

    overlap_panel[overlap_mask.astype(bool)] = (255, 0, 255)
    overlap_panel = annotate_image(
        overlap_panel,
        FrameAnnotation(
            bboxes=[],
            category=Action.NONE,
            confidence=100,
            reason="overlap",
        ),
        "overlap",
    )

    y = 60
    for line in _score_lines(overlap_scores):
        cv2.putText(
            overlap_panel,
            f"rnd-vs-pgt {line}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    preview = np.concatenate([null_preview, random_preview, pgt_preview, overlap_panel], axis=1)
    out_path = tmp_path / "sam3_polygon_e2e_preview.png"
    assert cv2.imwrite(str(out_path), preview)
    assert out_path.exists()
    print("preview_path=", out_path)


@pytest.mark.asyncio
async def test_sam3_polygon_end_to_end_polygons_only(tmp_path, monkeypatch):
    settings = get_settings()
    frame = _load_test_frame()
    pgt, gt_boxes = await _build_sam3_pgt(frame, monkeypatch)
    pseudo_gt = [pgt]

    null_predictions = {1: {"polygons": []}}
    pgt_predictions = {1: {"polygons": gt_boxes}}

    rng = np.random.default_rng(21)
    random_count = max(10, len(gt_boxes) * 8)
    random_polygons = [
        BoundingBox(
            bbox_2d=gt_boxes[0].bbox_2d,
            polygon=_overlap_polygon_from_box(gt_boxes[0], shrink=10),
            label=gt_boxes[0].label if gt_boxes else "player",
            cluster_id=ShirtColor.OTHER,
        )
    ] + [
        BoundingBox(
            bbox_2d=(0, 0, 10, 10),
            polygon=_make_random_polygon(
                rng,
                settings.SCOREVISION_IMAGE_WIDTH,
                settings.SCOREVISION_IMAGE_HEIGHT,
            ),
            label=gt_boxes[0].label if gt_boxes else "player",
            cluster_id=ShirtColor.OTHER,
        )
        for _ in range(random_count - 1)
    ]
    random_predictions = {1: {"polygons": random_polygons}}
    bbox_predictions = {1: {"bboxes": [_bbox_only_box(box) for box in random_polygons]}}

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
    bbox_scores = {
        "iou": compare_object_placement(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
        "map50": compare_map50(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
        "count": compare_object_counts(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
        "precision": compare_precision(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
        "recall": compare_recall(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
        "false_positive": compare_false_positive(pseudo_gt=pseudo_gt, miner_predictions=bbox_predictions),
    }

    assert pgt_scores["iou"] == pytest.approx(1.0)
    assert random_scores["iou"] > 0.0
    assert null_scores["iou"] == pytest.approx(0.0)
    assert bbox_scores["iou"] != pytest.approx(random_scores["iou"])
    assert bbox_scores["map50"] != pytest.approx(random_scores["map50"])
    assert bbox_scores["precision"] != pytest.approx(random_scores["precision"])
    assert bbox_scores["recall"] != pytest.approx(random_scores["recall"])

    null_preview = annotate_image(frame, FrameAnnotation(bboxes=[], category=Action.NONE, confidence=100, reason="null"), "null")
    random_preview = annotate_image(frame, FrameAnnotation(bboxes=[_to_polygon_box(box) for box in random_polygons], category=Action.NONE, confidence=100, reason="random"), "random")
    pgt_preview = annotate_image(frame, pgt.annotation, "pgt")

    overlap_panel = frame.copy()
    overlap_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for pred_box in random_polygons:
        pred_mask = _poly_mask(pred_box.polygon or [], frame.shape[:2])
        for gt_box in gt_boxes:
            gt_mask = _poly_mask(gt_box.polygon or [], frame.shape[:2])
            overlap_mask = np.maximum(overlap_mask, np.logical_and(pred_mask, gt_mask).astype(np.uint8))
    overlap_panel[overlap_mask.astype(bool)] = (255, 0, 255)
    overlap_panel = annotate_image(overlap_panel, FrameAnnotation(bboxes=[], category=Action.NONE, confidence=100, reason="overlap"), "overlap")

    y = 60
    for line in _score_lines(random_scores):
        cv2.putText(overlap_panel, f"rnd-vs-pgt {line}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    preview = np.concatenate([null_preview, random_preview, pgt_preview, overlap_panel], axis=1)
    out_path = tmp_path / "sam3_polygon_e2e_polygons_only_preview.png"
    assert cv2.imwrite(str(out_path), preview)
    assert out_path.exists()
    print("polygons_only_preview_path=", out_path)
