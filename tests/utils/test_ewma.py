import pytest

from scorevision.utils.ewma import (
    calculate_ewma_alpha,
    update_ewma_score,
    load_previous_ewma,
    save_ewma,
)
from scorevision.utils.prometheus import CACHE_DIR


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


def test_save_and_load_ewma(tmp_path, monkeypatch):
    monkeypatch.setattr("scorevision.utils.ewma.CACHE_DIR", tmp_path)

    scores = {"1": 0.5, "2": 0.8}
    save_ewma(10, scores)

    loaded = load_previous_ewma(10)
    assert loaded == scores


def test_ewma_first_window():
    alpha = calculate_ewma_alpha(3.0)
    s = update_ewma_score(current_score=0.7, previous_ewma=None, alpha=alpha)
    assert s == 0.7


def test_ewma_update_with_history():
    alpha = calculate_ewma_alpha(3)
    prev = 0.2
    cur = 0.8

    updated = update_ewma_score(cur, prev, alpha)
    assert updated == alpha * cur + (1 - alpha) * prev


def test_ewma_persistence_and_update(tmp_path, monkeypatch):
    monkeypatch.setattr("scorevision.utils.ewma.CACHE_DIR", tmp_path)

    # Previous window
    save_ewma(5, {"1": 0.4})

    prev = load_previous_ewma(5)
    alpha = calculate_ewma_alpha(3.0)

    updated = update_ewma_score(0.9, prev.get("1"), alpha)
    assert 0.4 < updated < 0.9  # in between

    save_ewma(6, {"1": updated})
    assert load_previous_ewma(6)["1"] == updated


def test_load_previous_ewma_corrupt_file(tmp_path, monkeypatch):
    """If the EWMA JSON is malformed, load_previous_ewma should return {} and not crash."""
    monkeypatch.setattr("scorevision.utils.ewma.CACHE_DIR", tmp_path)

    path = tmp_path / "ewma_42.json"
    path.write_text("{ this is not valid json }")

    scores = load_previous_ewma(42)
    assert scores == {}  # recovers gracefully


def test_update_ewma_multiple_miners(tmp_path, monkeypatch):
    """Ensure update_ewma_score works correctly for multiple miners/elements."""
    monkeypatch.setattr("scorevision.utils.ewma.CACHE_DIR", tmp_path)

    # previous window scores for 2 miners
    prev_scores = {"1": 0.2, "2": 0.5}
    save_ewma(1, prev_scores)

    alpha = calculate_ewma_alpha(3.0)
    current_scores = {"1": 0.8, "2": 0.6}

    ewma_scores = {
        uid: update_ewma_score(
            current_score=score, previous_ewma=prev_scores[uid], alpha=alpha
        )
        for uid, score in current_scores.items()
    }

    # Confirm each updated score is in-between prev and current
    assert 0.2 < ewma_scores["1"] < 0.8
    assert 0.5 < ewma_scores["2"] < 0.6


@pytest.mark.parametrize(
    "alpha, prev, current, expected",
    [
        (0.0, 0.5, 0.9, 0.5),  # α=0 → full smoothing, ignore current
        (1.0, 0.5, 0.9, 0.9),  # α=1 → immediate update
    ],
)
def test_update_ewma_alpha_boundary(alpha, prev, current, expected):
    """Test boundary alpha values α=0 and α=1 behave as expected."""
    updated = update_ewma_score(current_score=current, previous_ewma=prev, alpha=alpha)
    assert updated == pytest.approx(expected)
