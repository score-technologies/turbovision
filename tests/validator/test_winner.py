from scorevision.validator.winner import compute_adaptive_delta_rel


def test_compute_adaptive_delta_rel_with_baseline_theta():
    delta_rel = compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=0.3755112726134544)
    assert round(delta_rel, 6) == 0.056286


def test_compute_adaptive_delta_rel_clamps_to_bounds():
    assert compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=-1.0) == 0.03
    assert compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=2.0) == 0.10


def test_compute_adaptive_delta_rel_falls_back_to_default():
    assert compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=None) == 0.05
    assert compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=float("nan")) == 0.05
