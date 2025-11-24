import pytest

from scorevision.utils.ewma import calculate_ewma_alpha, update_ewma_score


def test_calculate_ewma_alpha_default_half_life():
    """h=3 should match the paper: α ≈ 0.2063."""
    alpha = calculate_ewma_alpha(3.0)
    assert alpha == pytest.approx(0.2063, rel=1e-3)


@pytest.mark.parametrize("h_small,h_large", [(1.0, 10.0), (0.5, 5.0)])
def test_alpha_monotonic_with_half_life(h_small, h_large):
    """
    Smaller half-life → larger alpha (faster adaptation).
    Larger half-life → smaller alpha (more smoothing).
    """
    alpha_small = calculate_ewma_alpha(h_small)
    alpha_large = calculate_ewma_alpha(h_large)

    assert 0.0 < alpha_small < 1.0
    assert 0.0 < alpha_large < 1.0
    assert alpha_small > alpha_large


@pytest.mark.parametrize("h", [0.1, 1.0, 3.0, 10.0, 100.0])
def test_alpha_bounds(h):
    """Alpha must always be in (0,1) for positive half-life."""
    alpha = calculate_ewma_alpha(h)
    assert 0.0 <= alpha <= 1.0


@pytest.mark.parametrize("h", [0.0, -1.0, -10.0])
def test_alpha_invalid_half_life_raises(h):
    with pytest.raises(AssertionError):
        calculate_ewma_alpha(h)


def test_update_ewma_no_history_returns_current():
    """If there is no previous EWMA, we start with the current score."""
    alpha = calculate_ewma_alpha(3.0)
    current_score = 0.7
    updated = update_ewma_score(
        current_score=current_score, previous_ewma=None, alpha=alpha
    )
    assert updated == pytest.approx(current_score)


def test_update_ewma_single_step():
    alpha = calculate_ewma_alpha(3.0)
    prev = 0.0
    current = 1.0
    updated = update_ewma_score(current_score=current, previous_ewma=prev, alpha=alpha)
    # S_t = α * 1 + (1-α) * 0 = α
    assert updated == pytest.approx(alpha)


def test_update_ewma_converges_to_constant():
    """
    Repeatedly feeding the same current_score should converge the EWMA
    toward that value.
    """
    alpha = calculate_ewma_alpha(3.0)
    current_score = 0.8
    ewma = 0.0  # start far away

    for _ in range(50):
        ewma = update_ewma_score(
            current_score=current_score, previous_ewma=ewma, alpha=alpha
        )

    assert ewma == pytest.approx(current_score, rel=1e-2)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_update_ewma_invalid_alpha_raises(alpha):
    with pytest.raises(AssertionError):
        update_ewma_score(current_score=0.5, previous_ewma=0.5, alpha=alpha)
