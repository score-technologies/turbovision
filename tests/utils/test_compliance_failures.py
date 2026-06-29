from scorevision.utils.compliance_failures import (
    ComplianceFailureTuple,
    is_compliance_tuple_failed,
    parse_compliance_failure_tuples,
)


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
