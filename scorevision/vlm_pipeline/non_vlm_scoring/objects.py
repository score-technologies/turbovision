from __future__ import annotations

from collections import Counter
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


# ===============================
# Generic helpers
# ===============================
def _iou_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


def _extract_boxes_labels(
    bboxes: Iterable[BoundingBox],
    *,
    only_players: bool = False,
    use_team: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], List[object]]:
    """
    Returns (boxes, labels) where:
      - boxes: [(x1,y1,x2,y2), ...]
      - labels: either class (ObjectOfInterest) or team (ShirtColor/TEAM1/TEAM2) depending on use_team
    """
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[object] = []
    for bb in bboxes or []:
        if only_players and bb.label != "player":
            continue
        boxes.append(tuple(bb.bbox_2d))
        labels.append(bb.cluster_id if use_team else bb.label)
    return boxes, labels


def _hungarian_f1(
    p_boxes: List[Tuple[int, int, int, int]],
    p_labels: List[object],
    h_boxes: List[Tuple[int, int, int, int]],
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
    p_boxes: List[Tuple[int, int, int, int]],
    p_labels: List[object],
    h_boxes: List[Tuple[int, int, int, int]],
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
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0


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
        per_frame.append(val)

    return float(sum(per_frame) / len(per_frame)) if per_frame else 0.0
