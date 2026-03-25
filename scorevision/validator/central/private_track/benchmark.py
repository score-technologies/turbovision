import logging
from typing import NamedTuple
import numpy as np
from scorevision.utils.actions import ACTION_CLASS_INDEX, NUM_ACTION_CLASSES
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.settings import get_settings

logger = logging.getLogger(__name__)


class BenchmarkResult(NamedTuple):
    map_at_1s: float
    per_action_ap: dict[str, float]


def compute_map_at_1s(
    predictions: list[FramePrediction],
    ground_truth: list[FramePrediction],
) -> BenchmarkResult:
    settings = get_settings()
    framerate = settings.PRIVATE_FRAME_RATE
    vector_size = settings.BENCHMARK_MAX_VIDEO_DURATION_MINUTES * 60 * framerate

    ground_truth_vector = _vectorize_ground_truth(ground_truth, vector_size)
    predictions_vector = _vectorize_predictions(predictions, vector_size)
    closest_vector = _build_closest_event_assignment(ground_truth_vector)

    delta_in_frames = 1 * framerate

    precision, recall = _compute_precision_recall_curve(
        ground_truth_vector,
        closest_vector,
        predictions_vector,
        delta_in_frames,
        settings.BENCHMARK_PRECISION_RECALL_THRESHOLDS,
    )

    overall_map, per_class_ap = _compute_mean_average_precision(
        precision, recall, settings.BENCHMARK_AP_INTERPOLATION_POINTS
    )

    per_action_ap = _map_class_indices_to_action_names(per_class_ap)

    return BenchmarkResult(map_at_1s=float(overall_map), per_action_ap=per_action_ap)


def _vectorize_ground_truth(
    ground_truth: list[FramePrediction], vector_size: int
) -> np.ndarray:
    vector = np.zeros((vector_size, NUM_ACTION_CLASSES))
    for gt in ground_truth:
        class_index = ACTION_CLASS_INDEX.get(gt.action)
        if class_index is None:
            continue
        frame = int(gt.frame)
        if 0 <= frame < vector_size:
            vector[frame][class_index] = 1
    return vector


def _vectorize_predictions(
    predictions: list[FramePrediction], vector_size: int
) -> np.ndarray:
    vector = np.zeros((vector_size, NUM_ACTION_CLASSES)) - 1
    for pred in predictions:
        class_index = ACTION_CLASS_INDEX.get(pred.action)
        if class_index is None:
            continue
        frame = int(pred.frame)
        if 0 <= frame < vector_size:
            vector[frame][class_index] = pred.confidence
    return vector


def _build_closest_event_assignment(ground_truth_vector: np.ndarray) -> np.ndarray:
    num_frames, num_classes = ground_truth_vector.shape
    closest = np.zeros(ground_truth_vector.shape) - 1

    for c in range(num_classes):
        event_indices = np.where(ground_truth_vector[:, c] != 0)[0].tolist()
        if not event_indices:
            continue
        event_indices.insert(0, -event_indices[0])
        event_indices.append(2 * num_frames)
        for i in range(1, len(event_indices) - 1):
            start = max(0, (event_indices[i - 1] + event_indices[i]) // 2)
            stop = min(num_frames, (event_indices[i] + event_indices[i + 1]) // 2)
            closest[start:stop, c] = ground_truth_vector[event_indices[i], c]

    return closest


def _compute_class_detections(
    target: np.ndarray,
    closest: np.ndarray,
    detection: np.ndarray,
    delta: int,
) -> tuple[np.ndarray, int]:
    gt_indices = np.where(target != 0)[0]
    pred_indices = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indices]

    detections_array = np.zeros((len(pred_indices), 3))
    detections_array[:, 0] = pred_scores
    detections_array[:, 2] = closest[pred_indices]

    matched_pred_indices: list[int] = []

    for gt_index in gt_indices:
        best_score = -1.0
        best_pred_position = None

        for position, (pred_index, pred_score) in enumerate(
            zip(pred_indices, pred_scores)
        ):
            if pred_index < gt_index - delta:
                continue
            if pred_index > gt_index + delta:
                break
            if (
                abs(int(pred_index) - int(gt_index)) <= delta // 2
                and pred_score > best_score
                and pred_index not in matched_pred_indices
            ):
                best_score = pred_score
                best_pred_position = position

        if best_pred_position is not None:
            detections_array[best_pred_position, 1] = 1
            matched_pred_indices.append(pred_indices[best_pred_position])

    return detections_array, len(gt_indices)


def _compute_precision_recall_curve(
    targets: np.ndarray,
    closests: np.ndarray,
    detections: np.ndarray,
    delta: int,
    num_thresholds: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = targets.shape[-1]
    thresholds = np.linspace(0, 1, num_thresholds)

    all_precision = []
    all_recall = []

    for c in range(num_classes):
        seed_detection = np.zeros((1, 3))
        seed_detection[0, 0] = -1
        class_detections = seed_detection
        total_gt_count = 0

        det, gt_count = _compute_class_detections(
            targets[:, c], closests[:, c], detections[:, c], delta
        )
        class_detections = np.append(class_detections, det, axis=0)
        total_gt_count += gt_count

        class_precision = []
        class_recall = []
        for threshold in thresholds:
            above_threshold = np.where(class_detections[:, 0] >= threshold)[0]
            true_positives = np.sum(class_detections[above_threshold, 1])
            precision = np.nan_to_num(true_positives / len(above_threshold))
            recall = np.nan_to_num(true_positives / total_gt_count)
            class_precision.append(precision)
            class_recall.append(recall)

        all_precision.append(class_precision)
        all_recall.append(class_recall)

    precision_array = np.array(all_precision).transpose()
    recall_array = np.array(all_recall).transpose()

    for i in range(num_classes):
        sort_order = np.argsort(recall_array[:, i])
        precision_array[:, i] = precision_array[sort_order, i]
        recall_array[:, i] = recall_array[sort_order, i]

    return precision_array, recall_array


def _compute_mean_average_precision(
    precision: np.ndarray, recall: np.ndarray, interpolation_points: int
) -> tuple[float, np.ndarray]:
    num_classes = precision.shape[-1]
    ap_per_class = np.array([0.0] * num_classes)

    for i in range(num_classes):
        for recall_threshold in np.arange(interpolation_points) / 10:
            recall_above = np.where(recall[:, i] >= recall_threshold)[0]
            precision_at_recall = precision[recall_above, i]
            max_precision = 0.0
            if precision_at_recall.shape[0] != 0:
                max_precision = np.max(precision_at_recall)
            ap_per_class[i] += max_precision

    ap_per_class = ap_per_class / interpolation_points
    mean_ap = float(np.mean(ap_per_class))

    return mean_ap, ap_per_class


def _map_class_indices_to_action_names(per_class_ap: np.ndarray) -> dict[str, float]:
    return {
        action_name: round(float(per_class_ap[class_index]), 6)
        for action_name, class_index in ACTION_CLASS_INDEX.items()
    }
