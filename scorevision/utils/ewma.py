"""Exponentially Weighted Moving Average (EWMA)"""

from logging import getLogger

logger = getLogger(__name__)


def calculate_ewma_alpha(half_life_windows: float) -> float:
    """
    Compute the EWMA decay factor α from a half-life measured in evaluation windows.

    Definition (Appendix C):
        α = 1 − 2^(-1/h)

    Where:
      - h > 0 is the half-life in windows
      - α ∈ (0, 1) for valid h

    Args:
        half_life_windows: Half-life h in evaluation windows (must be > 0).

    Returns:
        alpha: EWMA decay factor in the range (0, 1).

    Raises:
        AssertionError: If half_life_windows is not positive or alpha is out of bounds.
    """
    assert (
        half_life_windows > 0
    ), f"half_life_windows must be > 0, got {half_life_windows}"
    alpha = 1.0 - 2 ** (-1.0 / float(half_life_windows))
    return min(max(alpha, 0.0), 1.0)


def update_ewma_score(
    current_score: float,
    previous_ewma: float | None,
    alpha: float,
) -> float:
    """
    Update EWMA window score given the current per-window mean and the previous EWMA.

    Definition (Section 3.9 / 5.2.2):
        S_e,t = α × ClipMean_e,t + (1 − α) × S_e,t−1

    Behavior:
      - If previous_ewma is None, this is the first window:
          S_e,t = current_score
      - Otherwise, apply the EWMA update.

    Args:
        current_score: ClipMean_e,t for this window (already aggregated across clips).
        previous_ewma: S_e,t-1 from the previous window, or None if no history.
        alpha: EWMA decay factor α, must be within [0, 1].

    Returns:
        Updated EWMA score S_e,t.

    Raises:
        AssertionError: If alpha is outside [0, 1].
    """
    if previous_ewma is None:
        return current_score
    assert 0.0 <= alpha <= 1.0, f"EWMA alpha must be in [0,1], got {alpha}"
    return alpha * current_score + (1.0 - alpha) * previous_ewma
