import pytest

from scorevision.utils.manifest import Element


@pytest.mark.parametrize(
    "score, expected_gate",
    [
        (0.1, 0.0),  # below baseline_theta=0.3
        (0.3, 0.0),  # exactly at baseline
        (0.5, 0.2),  # above baseline
    ],
)
def test_apply_baseline_gate(dummy_detect_element, score, expected_gate):
    result = dummy_detect_element.apply_baseline_gate(score)
    assert result == expected_gate


@pytest.mark.parametrize(
    "score, expected_improvement",
    [
        (0.1, 0.05),  # below baseline → delta_floor applied
        (0.3, 0.05),  # at baseline → delta_floor
        (0.5, 0.2),  # above baseline → actual margin
    ],
)
def test_improvement(dummy_detect_element, score, expected_improvement):
    result = dummy_detect_element.improvement(score)
    assert result == expected_improvement


@pytest.mark.parametrize(
    "improvement, beta, expected_weighted",
    [
        (0.1, 1.0, 0.1),
        (0.2, 1.5, 0.3),
        (0.05, 0.0, 0.0),  # beta=0
    ],
)
def test_apply_difficulty_weight(
    dummy_detect_element, improvement, beta, expected_weighted
):
    dummy_detect_element.beta = beta
    result = dummy_detect_element.apply_difficulty_weight(improvement)
    assert result == pytest.approx(expected_weighted)


@pytest.mark.parametrize(
    "score, expected_weighted",
    [
        (0.1, 0.05),  # below baseline → delta_floor * beta
        (0.3, 0.05),  # at baseline
        (0.5, 0.2),  # above baseline
    ],
)
def test_weight_score(dummy_detect_element, score, expected_weighted):
    result = dummy_detect_element.weight_score(score)
    assert result == expected_weighted


def test_negative_score_clamps_to_floor(dummy_detect_element):
    result = dummy_detect_element.weight_score(-0.5)
    assert result == dummy_detect_element.delta_floor  # cannot go below delta_floor


def test_zero_delta_floor(dummy_detect_element):
    dummy_detect_element.delta_floor = 0.0
    assert dummy_detect_element.weight_score(0.1) == 0.0


def test_negative_score_clamped(dummy_detect_element):
    dummy_detect_element.baseline_theta = 0.0
    dummy_detect_element.delta_floor = 0.05
    assert dummy_detect_element.weight_score(-1.0) == 0.05


@pytest.mark.parametrize("beta, expected_multiplier", [(0.0, 0.0), (10.0, 10.0)])
def test_beta_extremes(dummy_detect_element, beta, expected_multiplier):
    dummy_detect_element.beta = beta
    improvement = dummy_detect_element.improvement(0.5)
    assert dummy_detect_element.apply_difficulty_weight(improvement) == pytest.approx(
        improvement * beta
    )


def test_zero_delta_floor(dummy_detect_element):
    dummy_detect_element.delta_floor = 0.0
    assert dummy_detect_element.improvement(0.0) == 0.0
