from scorevision.validator.winner import (
    _apply_recent_commit_initial_zero_filter,
    _drop_initial_zero_scores,
    _extract_sample_block,
    compute_adaptive_delta_rel,
)


def test_compute_adaptive_delta_rel_with_baseline_theta():
    delta_rel = compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=0.3755112726134544)
    assert round(delta_rel, 6) == 0.090694


def test_compute_adaptive_delta_rel_clamps_to_bounds():
    assert compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=-1.0) == 0.04
    assert compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=2.0) == 0.175


def test_compute_adaptive_delta_rel_falls_back_to_default():
    assert compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=None) == 0.05
    assert compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=float("nan")) == 0.05


def test_drop_initial_zero_scores_only_prefix():
    filtered, dropped = _drop_initial_zero_scores([0.0, 0.0, 0.8, 0.0, 1.0])
    assert filtered == [0.8, 0.0, 1.0]
    assert dropped == 2


def test_drop_initial_zero_scores_no_prefix_zero():
    filtered, dropped = _drop_initial_zero_scores([0.2, 0.0, 0.8])
    assert filtered == [0.2, 0.0, 0.8]
    assert dropped == 0


def test_drop_initial_zero_scores_caps_at_five():
    filtered, dropped = _drop_initial_zero_scores([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    assert filtered == [0.0, 0.8]
    assert dropped == 5


def test_extract_sample_block_prefers_payload_block():
    assert _extract_sample_block(
        {"block": 98},
        {"block": 100, "telemetry": {"block": 99}},
        {"block": 99},
    ) == 100


def test_extract_sample_block_uses_telemetry_block():
    assert _extract_sample_block({}, {"telemetry": {"block": "101"}}, {"block": "101"}) == 101


def test_extract_sample_block_falls_back_to_line_block():
    assert _extract_sample_block({"block": "102"}, {}, {}) == 102


def test_apply_recent_commit_initial_zero_filter_applies_only_to_recent_commits():
    samples_by_uid = {
        1: [(100, 0.0), (101, 0.0), (102, 0.8), (103, 1.0)],
        2: [(100, 0.0), (101, 0.6), (102, 0.7)],
        3: [(100, 0.0), (101, 0.0), (102, 0.9)],
        4: [(100, 0.2), (101, 0.0), (102, 0.9)],
    }
    uid_to_hk = {1: "hk1", 2: "hk2", 3: "hk3", 4: "hk4"}
    first_commit_block_by_hk = {
        "hk1": 95,
        "hk2": 95,
        "hk3": 10,
        "hk4": 95,
    }
    filtered, dropped = _apply_recent_commit_initial_zero_filter(
        samples_by_uid=samples_by_uid,
        uid_to_hk=uid_to_hk,
        first_commit_block_by_hk=first_commit_block_by_hk,
        max_block=103,
        recent_commit_blocks=10,
    )

    assert filtered[1] == [0.8, 1.0]
    assert dropped[1] == 2

    assert filtered[2] == [0.6, 0.7]
    assert dropped[2] == 1

    assert filtered[3] == [0.0, 0.0, 0.9]
    assert dropped[3] == 0

    assert filtered[4] == [0.2, 0.0, 0.9]
    assert dropped[4] == 0


def test_apply_recent_commit_initial_zero_filter_can_be_disabled_for_private_lane():
    filtered, dropped = _apply_recent_commit_initial_zero_filter(
        samples_by_uid={1: [(100, 0.0), (101, 0.0), (102, 0.8)]},
        uid_to_hk={1: "hk1"},
        first_commit_block_by_hk={"hk1": 95},
        max_block=102,
        recent_commit_blocks=10,
        enabled=False,
    )

    assert filtered[1] == [0.0, 0.0, 0.8]
    assert dropped[1] == 0


def test_apply_recent_commit_initial_zero_filter_orders_scores_by_block():
    filtered, dropped = _apply_recent_commit_initial_zero_filter(
        samples_by_uid={1: [(102, 0.8), (100, 0.0), (101, 0.0), (103, 0.0)]},
        uid_to_hk={1: "hk1"},
        first_commit_block_by_hk={"hk1": 95},
        max_block=103,
        recent_commit_blocks=10,
    )

    assert filtered[1] == [0.8, 0.0]
    assert dropped[1] == 2
