from __future__ import annotations

from collections import Counter
from logging import getLogger
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.pillar_metric_registry import register_metric
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import (
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
    BoundingBox,
    ShirtColor,
)

AUC_IOU_THRESHOLDS = (0.3, 0.5)
ENUM_IOU_THRESHOLD = 0.3

logger = getLogger(__name__)


# ===============================
# Generic helpers
# ===============================
def _iou_box(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        # fast path
        return 0.0
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalize_class_name(label: object) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()


def _safe_detection_score(detection: BoundingBox) -> float:
    value = getattr(detection, "score", None)
    if value is None:
        value = getattr(detection, "confidence", None)
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 1.0
    if not np.isfinite(score):
        return 1.0
    return score


def _average_precision(recalls: List[float], precisions: List[float]) -> float:
    if not recalls or not precisions or len(recalls) != len(precisions):
        return 0.0
    mrec = [0.0] + [float(v) for v in recalls] + [1.0]
    mpre = [0.0] + [float(v) for v in precisions] + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    area = 0.0
    for idx in range(len(mrec) - 1):
        delta = mrec[idx + 1] - mrec[idx]
        if delta > 0:
            area += delta * mpre[idx + 1]
    return float(area)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_per_image_rows(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> List[dict]:
    rows: List[dict] = []
    for pgt in pseudo_gt:
        frame_number = pgt.frame_number
        miner_frame = miner_predictions.get(frame_number) or {}
        pgt_boxes = pgt.annotation.bboxes or []
        miner_boxes = miner_frame.get("bboxes") or []

        gt_detections = []
        for box in pgt_boxes:
            class_name = _normalize_class_name(box.label)
            if not class_name:
                continue
            gt_detections.append({"class": class_name, "bbox": list(box.bbox_2d)})

        pred_detections = []
        for box in miner_boxes:
            class_name = _normalize_class_name(box.label)
            if not class_name:
                continue
            pred_detections.append(
                {
                    "class": class_name,
                    "bbox": list(box.bbox_2d),
                    "score": _safe_detection_score(box),
                }
            )

        rows.append(
            {
                "image_id": str(frame_number),
                "gt": gt_detections,
                "predictions": pred_detections,
            }
        )
    return rows


def _evaluate_detection_metrics_at_threshold(
    *,
    per_image: List[dict],
    iou_threshold: float,
) -> dict:
    gt_by_class_image: dict[str, dict[str, list[list[float]]]] = {}
    pred_by_class: dict[str, list[tuple[float, str, list[float]]]] = {}
    class_names: set[str] = set()

    for row in per_image:
        image_id = str(row.get("image_id", ""))
        gts = row.get("gt") or []
        preds = row.get("predictions") or []

        for det in gts:
            class_name = _normalize_class_name(det.get("class"))
            bbox = det.get("bbox")
            if (
                not class_name
                or not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue
            class_names.add(class_name)
            gt_by_class_image.setdefault(class_name, {}).setdefault(image_id, []).append(
                [float(v) for v in bbox]
            )

        for det in preds:
            class_name = _normalize_class_name(det.get("class"))
            bbox = det.get("bbox")
            if (
                not class_name
                or not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue
            class_names.add(class_name)
            pred_by_class.setdefault(class_name, []).append(
                (float(det.get("score", 1.0)), image_id, [float(v) for v in bbox])
            )

    per_class_ap: dict[str, float] = {}
    map_candidates: List[float] = []
    global_tp = 0
    global_fp = 0
    global_gt = 0

    for class_name in sorted(class_names):
        gt_images = gt_by_class_image.get(class_name, {})
        n_gt = sum(len(items) for items in gt_images.values())
        preds = sorted(
            pred_by_class.get(class_name, []),
            key=lambda row: row[0],
            reverse=True,
        )

        matched: dict[str, list[bool]] = {
            image_id: [False] * len(items) for image_id, items in gt_images.items()
        }
        tp_flags: List[int] = []
        fp_flags: List[int] = []

        for _score, image_id, pred_bbox in preds:
            gt_boxes = gt_images.get(image_id, [])
            best_idx = -1
            best_iou = 0.0
            for gt_idx, gt_bbox in enumerate(gt_boxes):
                if matched[image_id][gt_idx]:
                    continue
                iou = _iou_box(
                    (pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]),
                    (gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                matched[image_id][best_idx] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp_total = int(sum(tp_flags))
        fp_total = int(sum(fp_flags))
        global_tp += tp_total
        global_fp += fp_total
        global_gt += int(n_gt)

        if n_gt > 0:
            cum_tp = 0
            cum_fp = 0
            recalls: List[float] = []
            precisions: List[float] = []
            for tp_flag, fp_flag in zip(tp_flags, fp_flags):
                cum_tp += tp_flag
                cum_fp += fp_flag
                recalls.append(cum_tp / n_gt)
                denom = cum_tp + cum_fp
                precisions.append(cum_tp / denom if denom > 0 else 0.0)
            ap = _average_precision(recalls, precisions)
            map_candidates.append(ap)
            per_class_ap[class_name] = ap
        else:
            per_class_ap[class_name] = 0.0

    precision = (
        global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    )
    recall = global_tp / global_gt if global_gt > 0 else 0.0
    ffpi = global_fp / len(per_image) if per_image else 0.0

    return {
        "map": _mean(map_candidates),
        "precision": precision,
        "recall": recall,
        "ffpi": float(ffpi),
        "per_class_ap": per_class_ap,
    }


def _evaluate_detection_metrics(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict]
) -> dict:
    per_image = _build_per_image_rows(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )
    if not per_image:
        return {"map_50": 0.0, "precision": 0.0, "recall": 0.0, "false_positive": 0.0}

    at_50 = _evaluate_detection_metrics_at_threshold(
        per_image=per_image,
        iou_threshold=0.5,
    )
    return {
        "map_50": float(at_50["map"]),
        "precision": float(at_50["precision"]),
        "recall": float(at_50["recall"]),
        "false_positive": max(0.0, 1.0 - (float(at_50["ffpi"]) / 10.0)),
    }


def _extract_boxes_labels(
    bboxes: Iterable[BoundingBox],
    *,
    only_players: bool = False,
    use_team: bool = False,
) -> Tuple[List[Tuple[float, float, float, float]], List[object]]:
    """
    Returns (boxes, labels) where:
      - boxes: [(x1,y1,x2,y2), ...]
      - labels: either class (ObjectOfInterest) or team (ShirtColor/TEAM1/TEAM2) depending on use_team
    """
    boxes: List[Tuple[float, float, float, float]] = []
    labels: List[object] = []
    for bb in bboxes or []:
        if only_players and bb.label != "player":
            continue
        boxes.append(tuple(bb.bbox_2d))
        labels.append(bb.cluster_id if use_team else bb.label)
    return boxes, labels


def _hungarian_f1(
    p_boxes: List[Tuple[float, float, float, float]],
    p_labels: List[object],
    h_boxes: List[Tuple[float, float, float, float]],
    h_labels: List[object],
    *,
    iou_thresh: float,
    label_strict: bool,
) -> float:
    """
    F1 based on optimal matching (Hungarian).
    - label_strict=True => a pair counts as TP only if labels match.
    """
    if len(p_boxes) == 0 and len(h_boxes) == 0:
        return 1.0
    if len(p_boxes) == 0 or len(h_boxes) == 0:
        return 0.0

    N, M = len(p_boxes), len(h_boxes)
    # We maximize IoU by minimizing negative IoU
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            iou = _iou_box(p_boxes[i], h_boxes[j])
            if label_strict and (p_labels[i] != h_labels[j]):
                iou = 0.0
            cost[i, j] = -iou

    rows, cols = linear_sum_assignment(cost)
    TP = 0
    matched_h = set()
    matched_g = set()
    for r, c in zip(rows, cols):
        sim = -cost[r, c]
        if sim >= iou_thresh:
            TP += 1
            matched_h.add(c)
            matched_g.add(r)

    FP = M - len(matched_h)
    FN = N - len(matched_g)
    denom = 2 * TP + FP + FN
    return (2 * TP) / denom if denom > 0 else 1.0


def _auc_f1(
    p_boxes: List[Tuple[float, float, float, float]],
    p_labels: List[object],
    h_boxes: List[Tuple[float, float, float, float]],
    h_labels: List[object],
    thresholds: Iterable[float],
    *,
    label_strict: bool,
) -> float:
    vals = [
        _hungarian_f1(
            p_boxes,
            p_labels,
            h_boxes,
            h_labels,
            iou_thresh=t,
            label_strict=label_strict,
        )
        for t in thresholds
    ]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _team_auc_f1(
    p_bboxes: Iterable[BoundingBox],
    h_bboxes: Iterable[BoundingBox],
    thresholds: Iterable[float],
) -> float:
    """
    Players only. Find two dominant jersey colors on PGT side, then test
    both TEAM mappings (TEAM1->cA/TEAM2->cB and TEAM1->cB/TEAM2->cA).
    Return the better AUC-F1.
    """
    # PGT colors (ShirtColor.*)
    p_boxes, p_team = _extract_boxes_labels(p_bboxes, only_players=True, use_team=True)
    # Miner TEAM1/TEAM2 (cluster_id expected TEAM1_SHIRT_COLOUR / TEAM2_SHIRT_COLOUR)
    h_boxes, h_team = _extract_boxes_labels(h_bboxes, only_players=True, use_team=True)

    # Degenerate cases
    if not p_boxes and not h_boxes:
        return 1.0
    if not p_boxes or not h_boxes:
        return 0.0

    # Two dominant jersey colors on PGT
    top2 = [c for c, _ in Counter(p_team).most_common(2)]
    if len(top2) == 0:
        top2 = [ShirtColor.OTHER]
    if len(top2) == 1:
        top2.append(ShirtColor.OTHER)
    cA, cB = top2[0], top2[1]

    def map_labels(h_team_list, m1=True):
        mapped = []
        for t in h_team_list:
            if t == TEAM1_SHIRT_COLOUR:
                mapped.append(cA if m1 else cB)
            elif t == TEAM2_SHIRT_COLOUR:
                mapped.append(cB if m1 else cA)
            else:
                mapped.append(ShirtColor.OTHER)
        return mapped

    h_team_m1 = map_labels(h_team, m1=True)
    h_team_m2 = map_labels(h_team, m1=False)

    f1_m1 = _auc_f1(p_boxes, p_team, h_boxes, h_team_m1, thresholds, label_strict=True)
    f1_m2 = _auc_f1(p_boxes, p_team, h_boxes, h_team_m2, thresholds, label_strict=True)
    return max(f1_m1, f1_m2)


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.IOU),
    (ElementPrefix.OBJECT_DETECTION, PillarName.IOU),
)
def compare_object_placement(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    """
    Placement (label-agnostic AUC-F1).
    Frame-wise average of AUC-F1 between PGT and miner boxes using Hungarian,
    ignoring labels (pure geometry).
    """
    if not pseudo_gt:
        return 0.0

    per_frame = []
    for pgt in pseudo_gt:
        fr = pgt.frame_number
        miner = miner_predictions.get(fr) or {}
        h_bboxes = miner.get("bboxes") or []
        p_boxes, p_lab = _extract_boxes_labels(
            pgt.annotation.bboxes, only_players=False, use_team=False
        )
        h_boxes, h_lab = _extract_boxes_labels(
            h_bboxes, only_players=False, use_team=False
        )
        val = _auc_f1(
            p_boxes, p_lab, h_boxes, h_lab, AUC_IOU_THRESHOLDS, label_strict=False
        )
        logger.info(
            "[bbox][iou] frame=%s pgt=%s miner=%s score=%.6f",
            fr,
            len(p_boxes),
            len(h_boxes),
            val,
        )
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0


def compare_object_labels(
    pseudo_gt: List[PseudoGroundTruth],
    miner_predictions: dict[int, dict],
) -> float:
    """
    Categorization (label-strict AUC-F1).
    Same AUC-F1 computation, but a pair counts as TP only if classes match
    (player/ref/goalie/ball).
    """
    if not pseudo_gt:
        return 0.0

    per_frame = []
    for pgt in pseudo_gt:
        fr = pgt.frame_number
        miner = miner_predictions.get(fr) or {}
        h_bboxes = miner.get("bboxes") or []
        p_boxes, p_lab = _extract_boxes_labels(
            pgt.annotation.bboxes, only_players=False, use_team=False
        )
        h_boxes, h_lab = _extract_boxes_labels(
            h_bboxes, only_players=False, use_team=False
        )
        val = _auc_f1(
            p_boxes, p_lab, h_boxes, h_lab, AUC_IOU_THRESHOLDS, label_strict=True
        )
        logger.info(
            "[bbox][labels] frame=%s pgt=%s miner=%s score=%.6f",
            fr,
            len(p_boxes),
            len(h_boxes),
            val,
        )
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0


def compare_team_labels(
    pseudo_gt: List[PseudoGroundTruth],
    miner_predictions: dict[int, dict],
) -> float:
    """
    Team (players-only AUC-F1 with dynamic TEAM↔color mapping).
    Robust to TEAM1/TEAM2 flips by testing both mappings to the two dominant PGT colors.
    """
    if not pseudo_gt:
        return 0.0

    per_frame = []
    for pgt in pseudo_gt:
        fr = pgt.frame_number
        miner = miner_predictions.get(fr) or {}
        h_bboxes = miner.get("bboxes") or []
        val = _team_auc_f1(pgt.annotation.bboxes, h_bboxes, AUC_IOU_THRESHOLDS)
        logger.info(
            "[bbox][team] frame=%s pgt=%s miner=%s score=%.6f",
            fr,
            len(pgt.annotation.bboxes or []),
            len(h_bboxes),
            val,
        )
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.MAP50),
    (ElementPrefix.OBJECT_DETECTION, PillarName.MAP50),
)
def compare_map50(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    return _evaluate_detection_metrics(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )["map_50"]


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.PRECISION),
    (ElementPrefix.OBJECT_DETECTION, PillarName.PRECISION),
)
def compare_precision(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    return _evaluate_detection_metrics(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )["precision"]


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.RECALL),
    (ElementPrefix.OBJECT_DETECTION, PillarName.RECALL),
)
def compare_recall(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    return _evaluate_detection_metrics(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )["recall"]


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.FALSE_POSITIVE),
    (ElementPrefix.OBJECT_DETECTION, PillarName.FALSE_POSITIVE),
)
def compare_false_positive(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    return _evaluate_detection_metrics(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )["false_positive"]


@register_metric((ElementPrefix.PLAYER_DETECTION, PillarName.PALETTE))
def compare_palette(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    return compare_team_labels(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )


@register_metric((ElementPrefix.PLAYER_DETECTION, PillarName.ROLE))
def compare_object_and_team_labels(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    score1 = compare_object_labels(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )
    score2 = compare_team_labels(
        pseudo_gt=pseudo_gt, miner_predictions=miner_predictions
    )
    return (score1 + score2) / 2


@register_metric(
    (ElementPrefix.PLAYER_DETECTION, PillarName.COUNT),
    (ElementPrefix.OBJECT_DETECTION, PillarName.COUNT),
)
def compare_object_counts(
    pseudo_gt: List[PseudoGroundTruth], miner_predictions: dict[int, dict], **kwargs
) -> float:
    """
    Enumeration (F1 at IoU τ=0.3, label-agnostic).
    Emphasizes presence/count agreement with a lenient IoU threshold.
    """
    if not pseudo_gt:
        return 0.0

    per_frame = []
    for pgt in pseudo_gt:
        fr = pgt.frame_number
        miner = miner_predictions.get(fr) or {}
        h_bboxes = miner.get("bboxes") or []
        p_boxes, p_lab = _extract_boxes_labels(
            pgt.annotation.bboxes, only_players=False, use_team=False
        )
        h_boxes, h_lab = _extract_boxes_labels(
            h_bboxes, only_players=False, use_team=False
        )
        val = _hungarian_f1(
            p_boxes,
            p_lab,
            h_boxes,
            h_lab,
            iou_thresh=ENUM_IOU_THRESHOLD,
            label_strict=False,
        )
        logger.info(
            "[bbox][count] frame=%s pgt=%s miner=%s score=%.6f",
            fr,
            len(p_boxes),
            len(h_boxes),
            val,
        )
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0
