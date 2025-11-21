def apply_baseline_gate(score: float, baseline_theta: float) -> float:
    """
    Computes the positive margin above baseline
    """
    return max(score - baseline_theta, 0.0)


def calculate_improvement(
    score: float, baseline_theta: float, delta_floor: float
) -> float:
    """
    Compute improvement over baseline with a minimum floor.

    Semantic definition:
        improvement = max(score - baseline_theta, delta_floor)

    Guarantee:
      - Never returns a negative value.
      - Returned improvement >= delta_floor.

    Parameters
    ----------
    score : float
    baseline_theta : float
    delta_floor : float
        Minimum allowed improvement. Must be >= 0.

    Returns
    -------
    float
        Clamped improvement value.
    """
    if baseline_theta is None:
        raise ValueError("baseline_theta is required")
    if delta_floor is None:
        raise ValueError("delta_floor is required")
    if delta_floor < 0:
        raise ValueError("delta_floor must be non-negative")

    raw_margin = score - baseline_theta
    return max(raw_margin, delta_floor)


def apply_difficulty_weight(improvement: float, beta: float) -> float:
    """
    Apply the difficulty weighting.

        weighted = beta * improvement

    Parameters
    ----------
    improvement : float
        Non-negative improvement value.
    beta : float
        Difficulty scaling parameter.

    Returns
    -------
    float
        Weighted improvement score.
    """
    if beta is None:
        raise ValueError("beta is required")
    return beta * improvement
