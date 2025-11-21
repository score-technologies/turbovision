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
def test_apply_baseline_gate(sample_elements, score, expected_gate):
    element = sample_elements[0]  # PlayerDetect_v1 baseline_theta=0.3
    result = element.apply_baseline_gate(score)
    assert result == expected_gate


@pytest.mark.parametrize(
    "score, expected_improvement",
    [
        (0.1, 0.05),  # below baseline → delta_floor applied
        (0.3, 0.05),  # at baseline → delta_floor
        (0.5, 0.2),  # above baseline → actual margin
    ],
)
def test_improvement(sample_elements, score, expected_improvement):
    element = sample_elements[0]
    result = element.improvement(score)
    assert result == expected_improvement


@pytest.mark.parametrize(
    "improvement, beta, expected_weighted",
    [
        (0.1, 1.0, 0.1),
        (0.2, 1.5, 0.3),
        (0.05, 0.0, 0.0),  # beta=0
    ],
)
def test_apply_difficulty_weight(sample_elements, improvement, beta, expected_weighted):
    element = sample_elements[0]
    element.beta = beta
    result = element.apply_difficulty_weight(improvement)
    assert result == pytest.approx(expected_weighted)


@pytest.mark.parametrize(
    "score, expected_weighted",
    [
        (0.1, 0.05),  # below baseline → delta_floor * beta
        (0.3, 0.05),  # at baseline
        (0.5, 0.2),  # above baseline
    ],
)
def test_weight_score(sample_elements, score, expected_weighted):
    element = sample_elements[0]  # beta=1.0
    result = element.weight_score(score)
    assert result == expected_weighted


def test_negative_score_clamps_to_floor(sample_elements):
    element = sample_elements[0]
    result = element.weight_score(-0.5)
    assert result == element.delta_floor  # cannot go below delta_floor


def test_zero_delta_floor(sample_elements):
    element = sample_elements[0]
    element.delta_floor = 0.0
    assert element.weight_score(0.1) == 0.0
