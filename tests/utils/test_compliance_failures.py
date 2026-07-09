import os

import pytest

from scorevision.utils.compliance_failures import (
    ComplianceFailureTuple,
    fetch_compliance_failure_tuples,
    is_compliance_tuple_failed,
    parse_compliance_failure_tuples,
)


DEPLOYED_FAILING_TUPLES_URL = "https://conformity.scoredata.me/compliance/failing_tuples.json"


def test_parse_compliance_failure_tuples_from_public_shape():
    rows = [
        {
            "hotkey": " hk1 ",
            "element_id": "manak0/Detect-fire",
            "commit_block": "123",
            "latest_status": "FAIL_OUTPUT",
        },
        {"hotkey": "hk2", "element_id": "E2", "commit_block": None},
    ]

    assert parse_compliance_failure_tuples(rows) == {
        ComplianceFailureTuple("hk1", "manak0/Detect-fire", 123)
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
