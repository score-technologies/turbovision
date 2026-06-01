from decimal import Decimal, InvalidOperation
from typing import Callable

from scorevision.utils.actions import ACTION_CONFIGS, Action
from scorevision.utils.schemas import (
    CricketDeliveryPrediction,
    FramePrediction,
    SnookerBallPrediction,
    SnookerBallStateFrame,
    SnookerBallStatePrediction,
)
from scorevision.utils.settings import get_settings

PRIVATE_SCORING_VERSION = 4

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

_SNOOKER_UNIQUE_LABELS = {
    "cue",
    "yellow",
    "green",
    "brown",
    "blue",
    "pink",
    "black",
}
_SNOOKER_VALID_LABELS = _SNOOKER_UNIQUE_LABELS | {"red"}
_SNOOKER_VALID_STATES = {"on_table", "potted", "occluded", "unknown"}
_SNOOKER_COORDINATE_TOLERANCE = 0.05
_SNOOKER_DUPLICATE_FRAME_PENALTY = 0.10
_SNOOKER_COMPONENT_WEIGHTS = {
    "coordinate_accuracy": 0.45,
    "identity_accuracy": 0.25,
    "red_count_accuracy": 0.10,
    "state_accuracy": 0.15,
    "false_positive_score": 0.05,
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


def _normalize_snooker_label(label: str) -> str:
    value = " ".join(str(label or "").strip().lower().split())
    aliases = {
        "cue ball": "cue",
        "white": "cue",
        "white ball": "cue",
        "red ball": "red",
        "yellow ball": "yellow",
        "green ball": "green",
        "brown ball": "brown",
        "blue ball": "blue",
        "pink ball": "pink",
        "black ball": "black",
    }
    return aliases.get(value, value)


def _normalize_snooker_state(state: str) -> str:
    value = " ".join(str(state or "").strip().lower().split())
    return value


def _snooker_distance_score(
    prediction: SnookerBallPrediction,
    ground_truth: SnookerBallPrediction,
    *,
    tolerance: float = _SNOOKER_COORDINATE_TOLERANCE,
) -> float | None:
    if prediction.x is None or prediction.y is None:
        return None
    if ground_truth.x is None or ground_truth.y is None:
        return None
    distance = (
        (float(prediction.x) - float(ground_truth.x)) ** 2
        + (float(prediction.y) - float(ground_truth.y)) ** 2
    ) ** 0.5
    if tolerance <= 0:
        return 1.0 if distance == 0 else 0.0
    return max(0.0, 1.0 - min(1.0, distance / tolerance))


def _empty_snooker_breakdown() -> dict[str, float]:
    return {
        "coordinate_accuracy": 0.0,
        "identity_accuracy": 0.0,
        "red_count_accuracy": 0.0,
        "state_accuracy": 0.0,
        "false_positive_score": 0.0,
        "snooker_ball_state": 0.0,
    }


def _requires_snooker_coordinates(ball: SnookerBallPrediction) -> bool:
    return ball.state == "on_table"


def _has_snooker_coordinates(ball: SnookerBallPrediction) -> bool:
    return ball.x is not None and ball.y is not None


def _is_valid_snooker_prediction_ball(ball: SnookerBallPrediction) -> bool:
    if ball.label not in _SNOOKER_VALID_LABELS:
        return False
    if ball.state not in _SNOOKER_VALID_STATES:
        return False
    if _requires_snooker_coordinates(ball) and not _has_snooker_coordinates(ball):
        return False
    return True


def _is_valid_snooker_ground_truth_ball(ball: SnookerBallPrediction) -> bool:
    if ball.label not in _SNOOKER_VALID_LABELS:
        return False
    if ball.state not in _SNOOKER_VALID_STATES:
        return False
    if _requires_snooker_coordinates(ball) and not _has_snooker_coordinates(ball):
        return False
    return True


def _match_score_for_assignment(
    prediction: SnookerBallPrediction,
    ground_truth: SnookerBallPrediction,
) -> float:
    if _requires_snooker_coordinates(ground_truth):
        coordinate_score = _snooker_distance_score(prediction, ground_truth) or 0.0
        if coordinate_score <= 0.0:
            return 0.0
        if prediction.state == ground_truth.state:
            return coordinate_score
        return 0.5 * coordinate_score
    return 1.0 if prediction.state == ground_truth.state else 0.0


def _weighted_snooker_score(breakdown: dict[str, float]) -> float:
    score = 0.0
    for key, weight in _SNOOKER_COMPONENT_WEIGHTS.items():
        score += weight * breakdown.get(key, 0.0)
    return max(0.0, min(1.0, score))


def _linear_assignment(scores: list[list[float]]) -> list[tuple[int, int]]:
    if not scores or not scores[0]:
        return []

    def _solve_min_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
        row_count = len(cost)
        col_count = len(cost[0])
        potentials_row = [0.0] * (row_count + 1)
        potentials_col = [0.0] * (col_count + 1)
        matching = [0] * (col_count + 1)
        parent_col = [0] * (col_count + 1)

        for row in range(1, row_count + 1):
            matching[0] = row
            current_col = 0
            min_values = [float("inf")] * (col_count + 1)
            used = [False] * (col_count + 1)

            while True:
                used[current_col] = True
                current_row = matching[current_col]
                delta = float("inf")
                next_col = 0

                for col in range(1, col_count + 1):
                    if used[col]:
                        continue
                    candidate = (
                        cost[current_row - 1][col - 1]
                        - potentials_row[current_row]
                        - potentials_col[col]
                    )
                    if candidate < min_values[col]:
                        min_values[col] = candidate
                        parent_col[col] = current_col
                    if min_values[col] < delta:
                        delta = min_values[col]
                        next_col = col

                for col in range(col_count + 1):
                    if used[col]:
                        potentials_row[matching[col]] += delta
                        potentials_col[col] -= delta
                    else:
                        min_values[col] -= delta

                current_col = next_col
                if matching[current_col] == 0:
                    break

            while True:
                previous_col = parent_col[current_col]
                matching[current_col] = matching[previous_col]
                current_col = previous_col
                if current_col == 0:
                    break

        return [
            (matching[col] - 1, col - 1)
            for col in range(1, col_count + 1)
            if matching[col] != 0
        ]

    row_count = len(scores)
    col_count = len(scores[0])
    normalized_scores = [
        [max(0.0, min(1.0, score)) for score in row]
        for row in scores
    ]

    if row_count <= col_count:
        cost_matrix = [[1.0 - score for score in row] for row in normalized_scores]
        return _solve_min_assignment(cost_matrix)

    transposed_scores = [
        [normalized_scores[row][col] for row in range(row_count)]
        for col in range(col_count)
    ]
    cost_matrix = [[1.0 - score for score in row] for row in transposed_scores]
    return [(col, row) for row, col in _solve_min_assignment(cost_matrix)]


def _score_snooker_frame(
    prediction_frame: SnookerBallStateFrame | None,
    ground_truth_frame: SnookerBallStateFrame,
) -> dict[str, float]:
    pred_balls = prediction_frame.balls if prediction_frame is not None else []
    gt_balls = ground_truth_frame.balls
    if not gt_balls:
        return _empty_snooker_breakdown()
    if not pred_balls:
        return _empty_snooker_breakdown()

    normalized_pred = [
        ball.model_copy(
            update={
                "label": _normalize_snooker_label(ball.label),
                "state": _normalize_snooker_state(ball.state),
            }
        )
        for ball in pred_balls
    ]
    normalized_gt = [
        ball.model_copy(
            update={
                "label": _normalize_snooker_label(ball.label),
                "state": _normalize_snooker_state(ball.state),
            }
        )
        for ball in gt_balls
    ]
    normalized_gt = [
        ball for ball in normalized_gt
        if _is_valid_snooker_ground_truth_ball(ball)
    ]
    gt_count = len(normalized_gt)
    if gt_count == 0:
        return _empty_snooker_breakdown()

    invalid_pred_count = sum(
        1 for ball in normalized_pred if not _is_valid_snooker_prediction_ball(ball)
    )
    valid_pred = [
        (idx, ball)
        for idx, ball in enumerate(normalized_pred)
        if _is_valid_snooker_prediction_ball(ball)
    ]

    used_pred: set[int] = set()
    matches: list[tuple[SnookerBallPrediction, SnookerBallPrediction]] = []

    for gt in normalized_gt:
        if gt.label == "red":
            continue
        if gt.label not in _SNOOKER_UNIQUE_LABELS:
            continue
        best_idx = None
        best_score = -1.0
        for idx, pred in valid_pred:
            if idx in used_pred or pred.label != gt.label:
                continue
            candidate_score = _match_score_for_assignment(pred, gt)
            if candidate_score > best_score:
                best_idx = idx
                best_score = candidate_score
        if best_idx is not None and best_score > 0.0:
            used_pred.add(best_idx)
            matches.append((normalized_pred[best_idx], gt))

    gt_reds = [ball for ball in normalized_gt if ball.label == "red"]
    pred_reds = [
        (idx, ball)
        for idx, ball in valid_pred
        if ball.label == "red" and idx not in used_pred
    ]
    red_scores = [
        [_match_score_for_assignment(pred, gt) for _pred_idx, pred in pred_reds]
        for gt in gt_reds
    ]
    for red_idx, pred_pos in _linear_assignment(red_scores):
        if red_idx >= len(gt_reds) or pred_pos >= len(pred_reds):
            continue
        if red_scores[red_idx][pred_pos] <= 0.0:
            continue
        pred_idx, _pred = pred_reds[pred_pos]
        if pred_idx in used_pred:
            continue
        used_pred.add(pred_idx)
        matches.append((normalized_pred[pred_idx], gt_reds[red_idx]))

    coordinate_targets = [
        ball
        for ball in normalized_gt
        if _requires_snooker_coordinates(ball) and _has_snooker_coordinates(ball)
    ]
    coordinate_target_ids = {id(ball) for ball in coordinate_targets}
    coordinate_sum = 0.0
    for pred, gt in matches:
        if id(gt) not in coordinate_target_ids:
            continue
        coordinate_sum += _snooker_distance_score(pred, gt) or 0.0
    coordinate_accuracy = (
        coordinate_sum / len(coordinate_targets)
        if coordinate_targets
        else len(matches) / gt_count
    )
    identity_accuracy = len(matches) / gt_count
    state_accuracy = (
        sum(1.0 for pred, gt in matches if pred.state == gt.state) / gt_count
    )
    red_count_accuracy = 1.0
    valid_pred_red_count = sum(
        1 for _idx, ball in valid_pred
        if ball.label == "red"
    )
    if gt_reds:
        red_count_accuracy = max(
            0.0,
            1.0 - abs(valid_pred_red_count - len(gt_reds)) / len(gt_reds),
        )
    else:
        red_count_accuracy = 1.0 if valid_pred_red_count == 0 and matches else 0.0

    false_positive_count = invalid_pred_count + max(0, len(valid_pred) - len(matches))
    false_positive_score = max(0.0, 1.0 - false_positive_count / max(1, len(normalized_pred)))

    breakdown = {
        "coordinate_accuracy": max(0.0, min(1.0, coordinate_accuracy)),
        "identity_accuracy": max(0.0, min(1.0, identity_accuracy)),
        "red_count_accuracy": max(0.0, min(1.0, red_count_accuracy)),
        "state_accuracy": max(0.0, min(1.0, state_accuracy)),
        "false_positive_score": max(0.0, min(1.0, false_positive_score)),
    }
    breakdown["snooker_ball_state"] = _weighted_snooker_score(breakdown)
    return breakdown


def score_snooker_ball_state_with_breakdown(
    prediction: SnookerBallStatePrediction | None,
    ground_truth: SnookerBallStatePrediction,
    *,
    target_frames: list[int] | None = None,
) -> tuple[float, dict[str, float]]:
    if not ground_truth.frames:
        return 0.0, _empty_snooker_breakdown()

    gt_by_frame = {frame.frame: frame for frame in ground_truth.frames}
    if target_frames is None:
        frame_ids = [frame.frame for frame in ground_truth.frames]
    else:
        frame_ids = []
        seen: set[int] = set()
        for raw_frame in target_frames:
            try:
                frame_id = int(raw_frame)
            except (TypeError, ValueError):
                continue
            if frame_id < 0 or frame_id in seen:
                continue
            seen.add(frame_id)
            frame_ids.append(frame_id)

    if not frame_ids:
        return 0.0, _empty_snooker_breakdown()

    target_frame_set = set(frame_ids)
    pred_by_frame: dict[int, SnookerBallStateFrame] = {}
    duplicate_target_frame_count = 0
    for frame in prediction.frames if prediction is not None else []:
        if frame.frame not in target_frame_set:
            continue
        if frame.frame in pred_by_frame:
            duplicate_target_frame_count += 1
            pred_by_frame[frame.frame] = SnookerBallStateFrame(
                frame=frame.frame,
                balls=[*pred_by_frame[frame.frame].balls, *frame.balls],
            )
            continue
        pred_by_frame[frame.frame] = frame

    frame_breakdowns = [
        _score_snooker_frame(pred_by_frame.get(frame_id), gt_by_frame[frame_id])
        if frame_id in gt_by_frame
        else _empty_snooker_breakdown()
        for frame_id in frame_ids
    ]
    aggregate: dict[str, float] = {}
    for key in frame_breakdowns[0].keys():
        aggregate[key] = sum(frame[key] for frame in frame_breakdowns) / len(frame_breakdowns)
    if duplicate_target_frame_count:
        duplicate_penalty = max(
            0.0,
            1.0 - (_SNOOKER_DUPLICATE_FRAME_PENALTY * duplicate_target_frame_count),
        )
        aggregate["false_positive_score"] *= duplicate_penalty
        aggregate["snooker_ball_state"] = _weighted_snooker_score(aggregate) * duplicate_penalty
    else:
        aggregate["snooker_ball_state"] = _weighted_snooker_score(aggregate)
    score = aggregate.get("snooker_ball_state", 0.0)
    return max(0.0, min(1.0, score)), aggregate


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
