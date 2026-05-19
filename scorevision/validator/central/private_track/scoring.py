from decimal import Decimal, InvalidOperation
from typing import Callable

from scorevision.utils.actions import ACTION_CONFIGS, Action
from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction
from scorevision.utils.settings import get_settings

PRIVATE_SCORING_VERSION = 2

PillarScorer = Callable[[list[FramePrediction], list[FramePrediction]], float]


def frame_to_seconds(frame: int) -> float:
    return frame / get_settings().PRIVATE_FRAME_RATE


def calculate_time_decay(time_diff: float, tolerance: float, min_score: float) -> float:
    if time_diff > tolerance:
        return 0.0
    return 1 - (time_diff / tolerance) * (1 - min_score)


def find_best_match(
    prediction: FramePrediction,
    ground_truth: list[FramePrediction],
    used_indices: set[int],
) -> tuple[int | None, float]:
    pred_time = frame_to_seconds(prediction.frame)

    try:
        config = ACTION_CONFIGS[Action(prediction.action)]
    except (ValueError, KeyError):
        return None, 0.0

    best_idx = None
    best_decay = 0.0

    for i, gt in enumerate(ground_truth):
        if i in used_indices or gt.action != prediction.action:
            continue

        time_diff = abs(pred_time - frame_to_seconds(gt.frame))
        if time_diff <= config.tolerance_seconds:
            decay = calculate_time_decay(time_diff, config.tolerance_seconds, config.min_score)
            if decay > best_decay:
                best_decay = decay
                best_idx = i

    return best_idx, best_decay


def _legacy_score_predictions(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
) -> float:
    if not ground_truth:
        return 0.0

    gt_total_weight = 0.0
    for gt in ground_truth:
        try:
            gt_total_weight += ACTION_CONFIGS[Action(gt.action)].weight
        except (ValueError, KeyError):
            continue

    if gt_total_weight == 0:
        return 0.0

    predictions = sorted(predictions, key=lambda p: p.frame)

    matched_score = 0.0
    unmatched_penalty = 0.0
    used_indices: set[int] = set()

    for pred in predictions:
        try:
            config = ACTION_CONFIGS[Action(pred.action)]
        except (ValueError, KeyError):
            continue

        match_idx, time_decay = find_best_match(pred, ground_truth, used_indices)

        if match_idx is not None:
            used_indices.add(match_idx)
            matched_score += config.weight * time_decay
        else:
            unmatched_penalty += config.weight

    return max(min((matched_score - unmatched_penalty) / gt_total_weight, 1.0), 0.0)


def _normalize_pillar_weights(pillar_weights: dict[str, float] | None) -> dict[str, float]:
    if not pillar_weights:
        return {}
    out: dict[str, float] = {}
    for raw_key, raw_weight in pillar_weights.items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        out[key] = out.get(key, 0.0) + weight
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def _normalize_pillar_key(name: str) -> str:
    return str(name).strip()


def _soccer_action_scorer(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
) -> float:
    return _legacy_score_predictions(predictions, ground_truth)


_PILLAR_SCORERS: dict[str, PillarScorer] = {
    "soccer_action": _soccer_action_scorer,
    "private_legacy_score": _soccer_action_scorer,
}

_CRICKET_FIELD_WEIGHTS: dict[str, float] = {
    "match": 0.005,
    "matchid": 0.005,
    "inningsid": 0.02,
    "overid": 0.02,
    "ball_in_over": 0.02,
    "ballid": 0.01,
    "xlsx_overs": 0.01,
    "scorecard_overs": 0.01,
    "kph": 0.16,
    "bounce_x": 0.18,
    "stump_y": 0.14,
    "deviation": 0.1,
    "swing_angle": 0.08,
    "stump_z": 0.08,
    "release_y": 0.02,
    "release_z": 0.02,
    "bounce_y": 0.02,
    "impact_x": 0.02,
    "impact_y": 0.02,
    "impact_z": 0.02,
    "interception_distance": 0.02,
    "runs": 0.01,
    "wickets": 0.01,
}

_CRICKET_FIELD_TOLERANCES: dict[str, float] = {
    "kph": 3.0,
    "release_y": 0.15,
    "release_z": 0.15,
    "bounce_x": 0.25,
    "bounce_y": 0.15,
    "impact_x": 0.25,
    "impact_y": 0.15,
    "impact_z": 0.15,
    "interception_distance": 0.25,
    "stump_y": 0.12,
    "stump_z": 0.12,
    "swing_angle": 2.0,
    "deviation": 2.0,
}

_EXACT_MATCH_FIELDS = {
    "match",
    "matchid",
    "inningsid",
    "overid",
    "ball_in_over",
    "ballid",
    "xlsx_overs",
    "scorecard_overs",
    "runs",
    "wickets",
}


def register_pillar_scorer(pillar: str, scorer: PillarScorer) -> None:
    key = _normalize_pillar_key(pillar)
    if not key:
        raise ValueError("pillar must be a non-empty string")
    _PILLAR_SCORERS[key] = scorer


def _normalize_exact_value(value: object) -> str:
    normalized = " ".join(str(value).strip().lower().split())
    try:
        numeric = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return normalized
    if not numeric.is_finite():
        return normalized
    numeric_text = format(numeric.normalize(), "f")
    if "." in numeric_text:
        numeric_text = numeric_text.rstrip("0").rstrip(".")
    return numeric_text or "0"


def _score_exact_match(predicted: object, actual: object) -> float:
    if predicted is None or actual is None:
        return 0.0
    return 1.0 if _normalize_exact_value(predicted) == _normalize_exact_value(actual) else 0.0


def _score_numeric_match(predicted: object, actual: object, tolerance: float) -> float:
    if predicted is None or actual is None:
        return 0.0
    try:
        distance = abs(float(predicted) - float(actual))
    except (TypeError, ValueError):
        return 0.0
    if tolerance <= 0:
        return 1.0 if distance == 0 else 0.0
    if distance >= tolerance or abs(distance - tolerance) <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - (distance / tolerance))


def score_cricket_prediction_with_breakdown(
    prediction: CricketDeliveryPrediction | None,
    ground_truth: CricketDeliveryPrediction,
) -> tuple[float, dict[str, float]]:
    weighted_score = 0.0
    breakdown: dict[str, float] = {}

    for field_name, weight in _CRICKET_FIELD_WEIGHTS.items():
        predicted_value = getattr(prediction, field_name) if prediction is not None else None
        actual_value = getattr(ground_truth, field_name)
        if field_name in _EXACT_MATCH_FIELDS:
            field_score = _score_exact_match(predicted_value, actual_value)
        else:
            field_score = _score_numeric_match(
                predicted_value,
                actual_value,
                _CRICKET_FIELD_TOLERANCES[field_name],
            )
        breakdown[field_name] = field_score
        weighted_score += weight * field_score

    return max(0.0, min(1.0, weighted_score)), breakdown


def score_predictions_for_pillar(
    *,
    pillar: str,
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
) -> float:
    scorer = _PILLAR_SCORERS.get(_normalize_pillar_key(pillar))
    if scorer is None:
        return 0.0
    score = float(scorer(predictions, ground_truth))
    return max(0.0, min(1.0, score))


def score_predictions_with_breakdown(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
    pillar_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    normalized_weights = _normalize_pillar_weights(pillar_weights)
    if not normalized_weights:
        legacy_score = _legacy_score_predictions(predictions, ground_truth)
        return legacy_score, {"private_track_score": legacy_score}

    weighted_sum = 0.0
    breakdown: dict[str, float] = {}

    for pillar, weight in normalized_weights.items():
        pillar_score = score_predictions_for_pillar(
            pillar=pillar,
            predictions=predictions,
            ground_truth=ground_truth,
        )
        breakdown[pillar] = pillar_score
        weighted_sum += pillar_score * weight

    return max(0.0, min(1.0, weighted_sum)), breakdown


def score_predictions(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
    pillar_weights: dict[str, float] | None = None,
) -> float:
    score, _ = score_predictions_with_breakdown(
        predictions,
        ground_truth,
        pillar_weights=pillar_weights,
    )
    return score
