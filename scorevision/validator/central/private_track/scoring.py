from scorevision.utils.actions import ACTION_CONFIGS, Action
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.settings import get_settings

PRIVATE_SCORING_VERSION = 2


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


def score_predictions_with_breakdown(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
    pillar_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    legacy_score = _legacy_score_predictions(predictions, ground_truth)
    normalized_weights = _normalize_pillar_weights(pillar_weights)
    if not normalized_weights:
        return legacy_score, {"private_track_score": legacy_score}

    weighted_sum = 0.0
    breakdown: dict[str, float] = {}

    for pillar, weight in normalized_weights.items():
        # Primary private pillar: exact legacy private scoring.
        # Keep backward compatibility for older manifests.
        if pillar in ("soccer_action", "private_legacy_score"):
            pillar_score = legacy_score
        else:
            pillar_score = 0.0
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
