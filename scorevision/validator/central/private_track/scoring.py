from scorevision.utils.actions import ACTION_CONFIGS, Action
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.settings import get_settings


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


def score_predictions(
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
