import json
from collections import deque
from types import SimpleNamespace

import pytest

from scorevision.validator import winner as winner_mod
from scorevision.validator.central.private_track import registry as private_registry_mod
from scorevision.validator.winner import (
    _apply_recent_commit_initial_zero_challenge_filter,
    _apply_recent_commit_initial_zero_filter,
    _drop_initial_zero_challenge_rows,
    _drop_initial_zero_scores,
    _drop_zero_scores,
    _extract_sample_block,
    _extract_sample_commit_block,
    _select_representative_commit_block,
    compute_adaptive_delta_rel,
)


def test_compute_adaptive_delta_rel_with_baseline_theta():
    delta_rel = compute_adaptive_delta_rel(
        default_delta_rel=0.03, baseline_theta=0.3755112726134544
    )
    assert round(delta_rel, 6) == 0.090694


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old_ip", "old_port", "old_commit_block", "old_sample_block", "expected_old_score"),
    [
        ("0.0.0.0", 1, 210, 220, 0.45),
        ("0.0.0.0", 0, 210, 220, None),
        ("127.0.0.1", 8000, 150, 220, None),
        ("127.0.0.1", 8000, 210, 150, None),
        ("127.0.0.1", 8000, None, 220, None),
    ],
)
async def test_private_winner_requires_valid_commitment_and_current_registration_scores(
    monkeypatch, old_ip, old_port, old_commit_block, old_sample_block, expected_old_score
):
    settings = SimpleNamespace(
        SCOREVISION_NETUID=44,
        SCOREVISION_MECHID=1,
        VALIDATOR_FALLBACK_UID=6,
        SCOREVISION_WINDOW_TIEBREAK_ENABLE=False,
    )
    metagraph = SimpleNamespace(
        hotkeys=["old", "current"],
        axons=[
            SimpleNamespace(ip=old_ip, port=old_port),
            SimpleNamespace(ip="127.0.0.1", port=8001),
        ],
        block_at_registration=[200, 100],
    )

    def commit(block):
        return (
            block,
            json.dumps(
                {"role": "miner", "track": "private", "element_id": "football"}
            ),
        )

    class FakeSubtensor:
        async def metagraph(self, _netuid, mechid=None):
            return metagraph

        async def get_all_revealed_commitments(self, _netuid):
            return {
                "old": [commit(old_commit_block)] if old_commit_block is not None else [],
                "current": [commit(110)],
            }

    async def fake_subtensor():
        return FakeSubtensor()

    async def fake_indexes(_netuid):
        return {"validator": "index"}

    async def fake_dataset(*_args, **_kwargs):
        samples = [
            ("old", 0.9, old_sample_block),
            ("current", 0.5, 220),
        ]
        if old_ip == "0.0.0.0" and old_port == 1:
            samples.append(("old", 0.0, 230))
        for hotkey, score, block in samples:
            yield {
                "payload": {
                    "element_id": "football",
                    "lane": "private",
                    "block": block,
                    "telemetry": {"miner": {"hotkey": hotkey}},
                    "metrics": {"composite_score": score},
                }
            }

    async def fake_first_commit(_netuid, **_kwargs):
        raise AssertionError("private scoring must not fetch commitments a second time")

    monkeypatch.setattr(winner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(private_registry_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(winner_mod, "get_subtensor", fake_subtensor)
    monkeypatch.setattr(winner_mod, "get_validator_indexes_from_chain", fake_indexes)
    monkeypatch.setattr(winner_mod, "dataset_sv_multi", fake_dataset)
    monkeypatch.setattr(winner_mod, "_first_commit_block_by_miner", fake_first_commit)

    async def fake_inactive():
        return set()

    monkeypatch.setattr(private_registry_mod, "fetch_inactive_miner_tuples", fake_inactive)

    winner_uid, scores, _meta, rows = await winner_mod.get_winner_for_element(
        element_id="football",
        current_window_id="block-220",
        tail=220,
        m_min=1,
        lane="private",
    )

    assert winner_uid == 1
    assert scores == (
        {0: expected_old_score, 1: 0.5}
        if expected_old_score is not None
        else {1: 0.5}
    )
    assert [row["hotkey"] for row in rows] == (
        ["old", "current"] if expected_old_score is not None else ["current"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tiebreak_enabled", [False, True])
async def test_public_winner_excludes_expired_recovery_scores(
    monkeypatch, tiebreak_enabled
):
    tiebreak_calls = []
    async def fake_subtensor():
        return SimpleNamespace(
            metagraph=lambda _netuid, mechid=None: fake_metagraph()
        )

    async def fake_metagraph():
        return SimpleNamespace(hotkeys=["expired", "active"])

    async def fake_indexes(_netuid):
        return {"validator": "index"}

    async def fake_dataset(*_args, **_kwargs):
        _kwargs["index_keys_out"]["index"] = ["already-loaded"]
        for hotkey, score in (("expired", 0.99), ("active", 0.5)):
            yield {
                "payload": {
                    "element_id": "element-a",
                    "lane": "public",
                    "telemetry": {"miner": {"hotkey": hotkey}},
                    "metrics": {"composite_score": score},
                    "block": 900,
                }
            }

    async def fake_first_commit(_netuid, **kwargs):
        assert kwargs["index_keys_by_url"] == {"index": ["already-loaded"]}
        kwargs["expired_recovery_hotkeys"].add("expired")
        return {"expired": 100, "active": 100}

    async def fake_challenge_scores(**kwargs):
        assert kwargs["excluded_uids"] == {0}
        return {}

    def fake_pick_winner(winner_uid, **kwargs):
        tiebreak_calls.append(kwargs["candidate_uids"])
        return winner_uid

    monkeypatch.setattr(winner_mod, "get_settings", lambda: SimpleNamespace(
        SCOREVISION_NETUID=18,
        SCOREVISION_MECHID=1,
        VALIDATOR_FALLBACK_UID=99,
        SCOREVISION_WINDOW_TIEBREAK_ENABLE=tiebreak_enabled,
        SCOREVISION_WINDOW_K_PER_VALIDATOR=3,
        SCOREVISION_WINDOW_DELTA_ABS=0.003,
        SCOREVISION_WINDOW_DELTA_REL=0.03,
    ))
    monkeypatch.setattr(winner_mod, "get_subtensor", fake_subtensor)
    monkeypatch.setattr(winner_mod, "get_validator_indexes_from_chain", fake_indexes)
    monkeypatch.setattr(winner_mod, "dataset_sv_multi", fake_dataset)
    monkeypatch.setattr(winner_mod, "_first_commit_block_by_miner", fake_first_commit)
    monkeypatch.setattr(
        winner_mod,
        "collect_recent_challenge_scores_by_validator_miner",
        fake_challenge_scores,
    )
    monkeypatch.setattr(winner_mod, "pick_winner_with_tiebreak", fake_pick_winner)

    winner_uid, scores, _meta, rows = await winner_mod.get_winner_for_element(
        element_id="element-a",
        current_window_id="block-900",
        tail=900,
        m_min=1,
        lane="public",
    )

    assert winner_uid == 1
    assert scores == {1: 0.5}
    assert [row["hotkey"] for row in rows] == ["active"]
    assert tiebreak_calls == ([{1}] if tiebreak_enabled else [])


def test_compute_adaptive_delta_rel_clamps_to_bounds():
    assert (
        compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=-1.0) == 0.04
    )
    assert (
        compute_adaptive_delta_rel(default_delta_rel=0.03, baseline_theta=2.0) == 0.175
    )


def test_compute_adaptive_delta_rel_falls_back_to_default():
    assert (
        compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=None) == 0.05
    )
    assert (
        compute_adaptive_delta_rel(default_delta_rel=0.05, baseline_theta=float("nan"))
        == 0.05
    )


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


def test_drop_zero_scores_removes_at_most_two_zeros_anywhere():
    filtered, dropped = _drop_zero_scores([0.7, 0.0, 0.8, 0.0, 0.0, 0.9])

    assert filtered == [0.7, 0.8, 0.0, 0.9]
    assert dropped == 2


def test_drop_initial_zero_challenge_rows_only_removes_zero_prefix():
    filtered, dropped = _drop_initial_zero_challenge_rows(
        [
            ("c1", 0.0),
            ("c2", 0.0),
            ("c3", 0.8),
            ("c4", 0.0),
            ("c5", 0.9),
        ]
    )

    assert filtered == [("c3", 0.8), ("c4", 0.0), ("c5", 0.9)]
    assert dropped == 2


def test_drop_initial_zero_challenge_rows_caps_at_five():
    filtered, dropped = _drop_initial_zero_challenge_rows(
        [(f"c{i}", 0.0) for i in range(1, 7)] + [("c7", 0.8)]
    )

    assert filtered == [("c6", 0.0), ("c7", 0.8)]
    assert dropped == 5


def test_tiebreak_zero_filter_only_applies_to_recent_commitments():
    histories = {
        ("validator", 1): deque([("c1", 0.0), ("c2", 0.0), ("c3", 0.8)]),
        ("validator", 2): deque([("c1", 0.0), ("c2", 0.0), ("c3", 0.8)]),
    }

    filtered, dropped = _apply_recent_commit_initial_zero_challenge_filter(
        challenge_scores_by_validator_miner=histories,
        uid_to_hk={1: "recent-hk", 2: "old-hk"},
        first_commit_block_by_hk={"recent-hk": 95, "old-hk": 20},
        max_block=100,
        recent_commit_blocks=10,
    )

    assert list(filtered[("validator", 1)]) == [("c3", 0.8)]
    assert list(filtered[("validator", 2)]) == [
        ("c1", 0.0),
        ("c2", 0.0),
        ("c3", 0.8),
    ]
    assert dropped == 2


def test_tiebreak_zero_filter_keeps_zeros_when_commit_block_is_unknown():
    histories = {
        ("validator", 1): deque([("c1", 0.0), ("c2", 0.8)]),
    }

    filtered, dropped = _apply_recent_commit_initial_zero_challenge_filter(
        challenge_scores_by_validator_miner=histories,
        uid_to_hk={1: "unknown-hk"},
        first_commit_block_by_hk={},
        max_block=100,
        recent_commit_blocks=10,
    )

    assert list(filtered[("validator", 1)]) == [("c1", 0.0), ("c2", 0.8)]
    assert dropped == 0


def test_tiebreak_zero_filter_keeps_non_prefix_zeros_for_recent_commitments():
    histories = {
        ("validator", 1): deque([("c1", 0.8, 100), ("c2", 0.0, 101), ("c3", 0.9, 102)]),
    }

    filtered, dropped = _apply_recent_commit_initial_zero_challenge_filter(
        challenge_scores_by_validator_miner=histories,
        uid_to_hk={1: "recent-hk"},
        first_commit_block_by_hk={"recent-hk": 95},
        max_block=102,
        recent_commit_blocks=10,
    )

    assert list(filtered[("validator", 1)]) == [
        ("c1", 0.8, 100),
        ("c2", 0.0, 101),
        ("c3", 0.9, 102),
    ]
    assert dropped == 0


def test_extract_sample_block_prefers_payload_block():
    assert (
        _extract_sample_block(
            {"block": 98},
            {"block": 100, "telemetry": {"block": 99}},
            {"block": 99},
        )
        == 100
    )


def test_extract_sample_block_uses_telemetry_block():
    assert (
        _extract_sample_block({}, {"telemetry": {"block": "101"}}, {"block": "101"})
        == 101
    )


def test_extract_sample_block_falls_back_to_line_block():
    assert _extract_sample_block({"block": "102"}, {}, {}) == 102


def test_extract_sample_commit_block_from_shard_key():
    line = {
        "_key": (
            "https://example.com/manako/manak0_Detect-fire/hk1/"
            "000000123/evaluation/456-result.jsonl"
        )
    }

    assert _extract_sample_commit_block(line) == 123


def test_select_representative_commit_block_prefers_count_then_latest():
    assert _select_representative_commit_block({100: 2, 120: 2, 130: 1}) == 120


def test_public_zero_filter_applies_two_zero_allowance_to_old_and_recent_commits():
    samples_by_uid = {
        1: [(100, 0.0), (101, 0.0), (102, 0.8), (103, 1.0)],
        2: [(100, 0.0), (101, 0.6), (102, 0.7)],
        3: [(100, 0.0), (101, 0.0), (102, 0.9), (103, 0.0)],
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

    assert filtered[3] == [0.9, 0.0]
    assert dropped[3] == 2

    assert filtered[4] == [0.2, 0.9]
    assert dropped[4] == 1


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

    assert filtered[1] == [0.8]
    assert dropped[1] == 3


def test_apply_recent_commit_zero_filter_reduces_challenge_count_by_seven():
    filtered, dropped = _apply_recent_commit_initial_zero_filter(
        samples_by_uid={
            1: [
                (100, 0.0),
                (101, 0.0),
                (102, 0.0),
                (103, 0.0),
                (104, 0.0),
                (105, 0.8),
                (106, 0.0),
                (107, 0.9),
                (108, 0.0),
            ]
        },
        uid_to_hk={1: "hk1"},
        first_commit_block_by_hk={"hk1": 95},
        max_block=108,
        recent_commit_blocks=20,
    )

    assert filtered[1] == [0.8, 0.9]
    assert dropped[1] == 7
    assert len(filtered[1]) == 9 - dropped[1]
