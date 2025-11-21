from logging import getLogger

from scorevision.utils.manifest import Element

logger = getLogger(__name__)


def apply_baseline_gate(score: float, baseline_theta: float) -> float:
    """
    Computes the positive margin above baseline
    """
    return max(score - baseline_theta, 0.0)


def calculate_improvement(
    score: float, baseline_theta: float, delta_floor: float
) -> float:
    """
    Compute (non-negative) improvement over baseline with a minimum floor.
    """
    raw_margin = score - baseline_theta
    return max(raw_margin, delta_floor)


def apply_difficulty_weight(improvement: float, beta: float) -> float:
    """
    Apply the difficulty weighting.
    beta (Difficulty scaling parameter).
    """
    return beta * improvement


def weighted_score(score: float, manifest_element: Element) -> float:
    logger.info(f"Score (unweighted): {score}")
    # which score to pass in?
    # where does apply_baseline_gate fit in?
    improvement = calculate_improvement(
        score=score,
        baseline_theta=manifest_element.baseline_theta,
        delta_floor=manifest_element.delta_floor,
    )
    logger.info(f"Improvement: {improvement}")
    score_weighted = apply_difficulty_weight(
        improvement=improvement, beta=manifest_element.beta
    )
    logger.info(f"Score (weighted): {score_weighted}")
    return score_weighted
