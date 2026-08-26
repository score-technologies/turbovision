import asyncio

import pytest
from bittensor_wallet import Keypair

from scorevision.utils.run_signing import sign_run_payload
from scorevision.validator.audit.open_source import final_checker as fc


HK = "5DXmezvaGCwUEXKJ9sdZLwinFwMfdhqWy4GNC7me9gCPSdij"
ELEMENT = "manak0/Detect-car-wash"
BLOCK = 8868739
PREFIX = "compliance/runs/"


def _payload(ts: float, *rows: dict, prev: str | None = None) -> dict:
    return {
        "type": "public_compliance_run",
        "ts": ts,
        "prev_run_key": prev,
        "results": list(rows),
    }


def _row(status: str, p95: float = 115.0, *, hotkey: str = HK, block: int = BLOCK) -> dict:
    return {
        "hotkey": hotkey,
        "element_id": ELEMENT,
        "commit_block": block,
        "status": status,
        "p95_latency_ms": p95,
        "effective_latency_threshold_ms": 110.0,
    }


def _fold(*runs: tuple[str, dict], threshold: int = 3) -> list[dict]:
    tuples: dict = {}
    for key, payload in runs:
        tuples = fc.apply_run(tuples, key, payload, streak_threshold=threshold)
    return fc.failing_rows(tuples, streak_threshold=threshold)


def test_three_consecutive_breaches_ban():
    rows = _fold(
        ("runs/1.json", _payload(1.0, _row("PENDING_LATENCY"))),
        ("runs/2.json", _payload(2.0, _row("PENDING_LATENCY", 112.0))),
        ("runs/3.json", _payload(3.0, _row("PENDING_LATENCY", 118.0))),
    )

    assert len(rows) == 1
    assert rows[0]["latest_status"] == "FAIL_LATENCY"
    assert rows[0]["evidence_run_keys"] == ["runs/1.json", "runs/2.json", "runs/3.json"]


def test_two_breaches_are_not_enough():
    rows = _fold(
        ("runs/1.json", _payload(1.0, _row("PENDING_LATENCY"))),
        ("runs/2.json", _payload(2.0, _row("PENDING_LATENCY"))),
    )

    assert rows == []


def test_a_pass_in_between_breaks_the_streak():
    rows = _fold(
        ("runs/1.json", _payload(1.0, _row("PENDING_LATENCY"))),
        ("runs/2.json", _payload(2.0, _row("PENDING_LATENCY"))),
        ("runs/3.json", _payload(3.0, _row("PASS", 90.0))),
        ("runs/4.json", _payload(4.0, _row("PENDING_LATENCY"))),
    )

    assert rows == []


def test_a_later_pass_lifts_an_existing_ban():
    rows = _fold(
        ("runs/1.json", _payload(1.0, _row("PENDING_LATENCY"))),
        ("runs/2.json", _payload(2.0, _row("PENDING_LATENCY"))),
        ("runs/3.json", _payload(3.0, _row("PENDING_LATENCY"))),
        ("runs/4.json", _payload(4.0, _row("PASS", 88.0))),
    )

    assert rows == []


def test_output_failure_bans_immediately_and_a_pass_clears_it():
    banned = _fold(("runs/1.json", _payload(1.0, _row("FAIL_OUTPUT"))))
    assert banned[0]["latest_status"] == "FAIL_OUTPUT"

    cleared = _fold(
        ("runs/1.json", _payload(1.0, _row("FAIL_OUTPUT"))),
        ("runs/2.json", _payload(2.0, _row("PASS", 90.0))),
    )
    assert cleared == []


def test_checker_side_failures_never_ban():
    """SKIP and FAIL_RUNTIME describe our own blind spots, not miner behaviour."""
    base = {"hotkey": HK, "element_id": ELEMENT, "commit_block": BLOCK}
    rows = _fold(
        ("runs/1.json", _payload(1.0, {**base, "status": "SKIP"})),
        ("runs/2.json", _payload(2.0, {**base, "status": "FAIL_RUNTIME"})),
        ("runs/3.json", _payload(3.0, {**base, "status": "FAIL_RUNTIME"})),
    )

    assert rows == []


def test_a_claimed_breach_under_the_threshold_is_not_one():
    rows = _fold(
        ("runs/1.json", _payload(1.0, _row("FAIL_LATENCY", 95.0))),
        ("runs/2.json", _payload(2.0, _row("FAIL_LATENCY", 96.0))),
        ("runs/3.json", _payload(3.0, _row("FAIL_LATENCY", 97.0))),
    )

    assert rows == []


def test_a_ban_survives_the_tuple_leaving_the_targets():
    """A banned tuple stops being targeted; its verdict must not expire on its own."""
    tuples: dict = {}
    for i in (1, 2, 3):
        tuples = fc.apply_run(
            tuples, f"runs/{i}.json", _payload(float(i), _row("PENDING_LATENCY")), streak_threshold=3
        )
    assert len(fc.failing_rows(tuples, streak_threshold=3)) == 1

    for i in range(4, 60):
        tuples = fc.apply_run(
            tuples,
            f"runs/{i}.json",
            _payload(float(i), _row("PASS", 90.0, hotkey="someone-else")),
            streak_threshold=3,
        )

    assert [r["hotkey"] for r in fc.failing_rows(tuples, streak_threshold=3)] == [HK]


def test_cursor_only_moves_forward():
    """Re-inserting an old key in the mutable index must not replay it."""
    index = [f"{PREFIX}00000000{i}.json" for i in (1, 2, 3)]

    assert fc._new_keys(index, None) == index
    assert fc._new_keys(index, f"{PREFIX}000000002.json") == [f"{PREFIX}000000003.json"]
    # an attacker appends an already-processed key at the end of the index
    tampered = index + [f"{PREFIX}000000001.json"]
    assert fc._new_keys(tampered, f"{PREFIX}000000003.json") == []


def test_forged_and_unsigned_runs_are_rejected(monkeypatch):
    keypair = Keypair.create_from_uri("//RunSigner")
    attacker = Keypair.create_from_uri("//Attacker")
    key = f"{PREFIX}000000001.json"
    payload = _payload(1.0, _row("PENDING_LATENCY"))

    monkeypatch.setattr(
        fc,
        "get_settings",
        lambda: type("S", (), {"LATENCY_LOOP_HOTKEY": keypair.ss58_address})(),
    )

    assert fc._is_authentic(key, sign_run_payload(payload, run_key=key, keypair=keypair))
    assert not fc._is_authentic(key, sign_run_payload(payload, run_key=key, keypair=attacker))
    assert not fc._is_authentic(key, payload)


def test_signature_check_is_skipped_until_the_hotkey_is_pinned(monkeypatch):
    monkeypatch.setattr(fc, "get_settings", lambda: type("S", (), {"LATENCY_LOOP_HOTKEY": ""})())

    assert fc._is_authentic("runs/1.json", _payload(1.0, _row("PASS", 90.0)))


def test_a_hidden_run_breaks_the_chain():
    """The runs index is mutable: dropping an entry must not pass unnoticed."""
    cursor = f"{PREFIX}000000002.json"

    following = _payload(3.0, _row("PASS", 90.0), prev=cursor)
    assert fc.chain_is_continuous(following, cursor)

    # run 2 was removed from the index before we read it, so run 3 names a
    # predecessor we never saw
    orphan = _payload(3.0, _row("PASS", 90.0), prev=f"{PREFIX}000000009.json")
    assert not fc.chain_is_continuous(orphan, cursor)

    # nothing to compare against on a first run, or on an unchained payload
    assert fc.chain_is_continuous(orphan, None)
    assert fc.chain_is_continuous(_payload(3.0, _row("PASS", 90.0)), cursor)


class _FakeS3:
    """Minimal ListObjectsV2 stand-in that honours StartAfter and pagination."""

    def __init__(self, keys: list[str]):
        self.keys = sorted(keys)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def list_objects_v2(self, **params):
        self.calls.append(params)
        keys = [k for k in self.keys if k.startswith(params["Prefix"])]
        after = params.get("ContinuationToken") or params.get("StartAfter")
        if after:
            keys = [k for k in keys if k > after]
        page, rest = keys[: params["MaxKeys"]], keys[params["MaxKeys"] :]
        return {
            "Contents": [{"Key": k} for k in page],
            "IsTruncated": bool(rest),
            "NextContinuationToken": page[-1] if rest else None,
        }


def _fake_cfg():
    from scorevision.utils.r2 import R2Config

    return R2Config(bucket="conformity", account_id="a", access_key_id="b", secret_access_key="c", concurrency=1)


def test_listing_returns_only_runs_after_the_cursor(monkeypatch):
    keys = [f"{PREFIX}00000000{i}.json" for i in range(1, 6)] + [f"{PREFIX}index.json"]
    fake = _FakeS3(keys)
    monkeypatch.setattr(fc, "create_s3_client", lambda cfg, error_message: fake)
    monkeypatch.setattr(fc, "_runs_prefix", lambda: PREFIX)

    found = asyncio.run(fc._list_run_keys(_fake_cfg(), f"{PREFIX}000000003.json"))

    assert found == [f"{PREFIX}000000004.json", f"{PREFIX}000000005.json"]
    assert fake.calls[0]["StartAfter"] == f"{PREFIX}000000003.json"


def test_listing_paginates_and_skips_the_index(monkeypatch):
    keys = [f"{PREFIX}{i:09d}.json" for i in range(1, 2101)] + [f"{PREFIX}index.json"]
    fake = _FakeS3(keys)
    monkeypatch.setattr(fc, "create_s3_client", lambda cfg, error_message: fake)
    monkeypatch.setattr(fc, "_runs_prefix", lambda: PREFIX)

    found = asyncio.run(fc._list_run_keys(_fake_cfg(), None))

    assert len(found) == 2100
    assert all("index" not in k for k in found)
    assert len(fake.calls) == 3  # 1000 + 1000 + 100


def test_only_well_formed_run_keys_are_considered(monkeypatch):
    """The lock forbids overwriting, not creating: junk under the prefix is ignored."""
    junk = [
        f"{PREFIX}0deadbeef.json",
        f"{PREFIX}000000001.txt",
        f"{PREFIX}nested/000000001.json",
        f"{PREFIX}index.json",
        f"{PREFIX}000000001.json.bak",
    ]
    fake = _FakeS3(junk + [f"{PREFIX}000000001.json"])
    monkeypatch.setattr(fc, "create_s3_client", lambda cfg, error_message: fake)
    monkeypatch.setattr(fc, "_runs_prefix", lambda: PREFIX)

    assert asyncio.run(fc._list_run_keys(_fake_cfg(), None)) == [f"{PREFIX}000000001.json"]
