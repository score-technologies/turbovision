import os

import pytest

from scorevision.utils.compliance_failures import (
    ComplianceFailureTuple,
    fetch_compliance_failure_tuples,
    is_compliance_tuple_failed,
    parse_compliance_failure_tuples,
)


DEPLOYED_FAILING_TUPLES_URL = "https://turbo.scoredata.me/manako/conformity/failing_tuples.json"


def test_parse_compliance_failure_tuples_from_public_shape():
    rows = [
        {
            "hotkey": " hk1 ",
            "element_id": "manak0/Detect-fire",
            "commit_block": "123",
            "latest_status": "FAIL_OUTPUT",
            "latest_run_key": "compliance/runs/008595150.json",
        },
        {
            "hotkey": "hk2",
            "element_id": "E2",
            "commit_block": None,
            "latest_run_key": "compliance/runs/008595150.json",
        },
    ]

    assert parse_compliance_failure_tuples(rows) == {
        ComplianceFailureTuple("hk1", "manak0/Detect-fire", 123)
    }


def test_parse_compliance_failure_tuples_ignores_rows_without_run_key():
    rows = [
        {
            "hotkey": "hk1",
            "element_id": "manak0/Detect-fire",
            "commit_block": 123,
            "latest_status": "FAIL_OUTPUT",
        },
        {
            "hotkey": "hk2",
            "element_id": "manak0/Detect-fire",
            "commit_block": 124,
            "latest_status": "FAIL_OUTPUT",
            "latest_run_key": "   ",
        },
        {
            "hotkey": "hk3",
            "element_id": "manak0/Detect-fire",
            "commit_block": 125,
            "latest_status": "FAIL_LATENCY",
            "latest_run_key": "compliance/runs/008595150.json",
        },
    ]

    assert parse_compliance_failure_tuples(rows) == {
        ComplianceFailureTuple("hk3", "manak0/Detect-fire", 125)
    }


def test_is_compliance_tuple_failed_matches_exact_trio_only():
    failures = {ComplianceFailureTuple("hk1", "E1", 123)}

    assert is_compliance_tuple_failed(
        failures,
        hotkey="hk1",
        element_id="E1",
        commit_block=123,
    )
    assert not is_compliance_tuple_failed(
        failures,
        hotkey="hk1",
        element_id="E2",
        commit_block=123,
    )
    assert not is_compliance_tuple_failed(
        failures,
        hotkey="hk1",
        element_id="E1",
        commit_block=124,
    )


@pytest.mark.asyncio
@pytest.mark.deployment
@pytest.mark.skipif(
    os.getenv("SCOREVISION_RUN_DEPLOYMENT_TESTS") != "1",
    reason="set SCOREVISION_RUN_DEPLOYMENT_TESTS=1 to hit deployed compliance URL",
)
async def test_fetch_compliance_failure_tuples_from_deployed_url():
    failures = await fetch_compliance_failure_tuples(
        DEPLOYED_FAILING_TUPLES_URL,
        timeout_s=15.0,
        use_cache=False,
    )

    assert failures
    assert all(item.hotkey and item.element_id and item.commit_block >= 0 for item in failures)
    assert any(item.element_id.startswith("manak0/") for item in failures)


def test_a_stale_conformity_url_in_the_env_is_refused(monkeypatch):
    """That file is no longer written, and the conformity key can still write it.

    A validator left pointing at it would be reading an attacker-writable ban list,
    so the override is refused (and a warning is logged).
    """
    from scorevision.utils import settings as settings_mod

    monkeypatch.setenv(
        "SCOREVISION_FAILING_TUPLES_URL",
        "https://conformity.scoredata.me/compliance/failing_tuples.json",
    )

    assert settings_mod._failing_tuples_url() == settings_mod.DEFAULT_FAILING_TUPLES_URL


def test_an_explicit_url_is_still_honoured(monkeypatch):
    from scorevision.utils import settings as settings_mod

    monkeypatch.setenv("SCOREVISION_FAILING_TUPLES_URL", "https://example.test/mine.json")

    assert settings_mod._failing_tuples_url() == "https://example.test/mine.json"


def test_no_env_falls_back_to_the_owner_bucket(monkeypatch):
    from scorevision.utils import settings as settings_mod

    monkeypatch.delenv("SCOREVISION_FAILING_TUPLES_URL", raising=False)

    assert settings_mod._failing_tuples_url() == settings_mod.DEFAULT_FAILING_TUPLES_URL
