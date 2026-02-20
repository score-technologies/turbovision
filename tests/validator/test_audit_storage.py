from json import loads
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from scorevision.validator.audit.open_source.storage import (
    commit_audit_index_on_start,
    emit_spotcheck_result_shard,
)
from scorevision.validator.models import ChallengeRecord, SpotcheckResult


class _Secret:
    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _Client:
    def __init__(self):
        self.calls: list[dict] = []

    async def put_object(self, **kwargs):
        self.calls.append(kwargs)


class _ClientContext:
    def __init__(self, client: _Client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _record() -> ChallengeRecord:
    return ChallengeRecord(
        challenge_id="50414",
        element_id="PlayerDetect_v1@1.0",
        window_id="2025-10-27",
        block=7471716,
        miner_hotkey="5FsREvyUXSZWYRqVyQLDdpYmZZPnkhZyW6HjooozKP1nQkwu",
        central_score=0.819,
        payload={},
        responses_key="scorevision/PlayerDetect_v1@1.0/x/responses/007471716-50414.json",
        scored_frame_numbers=[10, 20, 30],
    )


def _result() -> SpotcheckResult:
    return SpotcheckResult(
        challenge_id="50414",
        element_id="PlayerDetect_v1@1.0",
        miner_hotkey="5FsREvyUXSZWYRqVyQLDdpYmZZPnkhZyW6HjooozKP1nQkwu",
        central_score=0.819,
        audit_score=0.801,
        match_percentage=0.978,
        passed=True,
        details={"n_pseudo_gt_frames": 3},
    )


@pytest.mark.asyncio
async def test_emit_spotcheck_result_shard_skips_without_audit_r2():
    settings = SimpleNamespace(
        SCOREVISION_VERSION="0.2.0",
        SCOREVISION_BUCKET="",
        AUDIT_R2_BUCKET="",
        AUDIT_R2_ACCOUNT_ID=_Secret(""),
        AUDIT_R2_WRITE_ACCESS_KEY_ID=_Secret(""),
        AUDIT_R2_WRITE_SECRET_ACCESS_KEY=_Secret(""),
        CENTRAL_R2_ACCOUNT_ID=_Secret(""),
        CENTRAL_R2_WRITE_ACCESS_KEY_ID=_Secret(""),
        CENTRAL_R2_WRITE_SECRET_ACCESS_KEY=_Secret(""),
        AUDIT_R2_CONCURRENCY=8,
        CENTRAL_R2_CONCURRENCY=8,
        AUDIT_R2_RESULTS_PREFIX="audit_spotcheck",
    )
    with patch("scorevision.validator.audit.open_source.storage.get_settings", return_value=settings):
        key = await emit_spotcheck_result_shard(
            record=_record(),
            result=_result(),
            mode="once",
            source="mock",
            threshold=0.95,
            tail_blocks=28800,
            mock_data_dir="tests/test_data/mock_spotcheck",
        )
    assert key is None


@pytest.mark.asyncio
async def test_emit_spotcheck_result_shard_uploads_payload():
    client = _Client()
    settings = SimpleNamespace(
        SCOREVISION_VERSION="0.2.0",
        SCOREVISION_BUCKET="central-bucket",
        AUDIT_R2_BUCKET="audit-bucket",
        AUDIT_R2_ACCOUNT_ID=_Secret("acc"),
        AUDIT_R2_WRITE_ACCESS_KEY_ID=_Secret("key"),
        AUDIT_R2_WRITE_SECRET_ACCESS_KEY=_Secret("secret"),
        CENTRAL_R2_ACCOUNT_ID=_Secret(""),
        CENTRAL_R2_WRITE_ACCESS_KEY_ID=_Secret(""),
        CENTRAL_R2_WRITE_SECRET_ACCESS_KEY=_Secret(""),
        AUDIT_R2_CONCURRENCY=8,
        CENTRAL_R2_CONCURRENCY=8,
        AUDIT_R2_RESULTS_PREFIX="audit_spotcheck",
    )
    with patch("scorevision.validator.audit.open_source.storage.get_settings", return_value=settings), \
         patch("scorevision.validator.audit.open_source.storage.ensure_audit_index_exists", new=AsyncMock(return_value=True)), \
         patch("scorevision.validator.audit.open_source.storage._audit_index_add_if_new", new=AsyncMock()), \
         patch("scorevision.validator.audit.open_source.storage._sign_batch", new=AsyncMock(return_value=("5Audit", ["0xsig"]))), \
         patch("scorevision.validator.audit.open_source.storage._get_audit_s3_client", return_value=_ClientContext(client)):
        key = await emit_spotcheck_result_shard(
            record=_record(),
            result=_result(),
            mode="once",
            source="mock",
            threshold=0.95,
            tail_blocks=28800,
            mock_data_dir="tests/test_data/mock_spotcheck",
        )

    assert key is not None
    assert "/spotcheck/" in key
    assert len(client.calls) == 1
    body = loads(client.calls[0]["Body"])
    assert isinstance(body, list) and len(body) == 1
    line = body[0]
    assert line["hotkey"] == "5Audit"
    assert line["signature"] == "0xsig"
    assert line["payload"]["result"]["passed"] is True
    assert line["payload"]["scored_frame_numbers"] == [10, 20, 30]


@pytest.mark.asyncio
async def test_commit_audit_index_on_start_skip_env(monkeypatch):
    monkeypatch.setenv("AUDIT_COMMIT_VALIDATOR_ON_START", "0")
    with patch("scorevision.validator.audit.open_source.storage._commit_audit_index", new=AsyncMock()) as commit_mock:
        await commit_audit_index_on_start()
    commit_mock.assert_not_called()


@pytest.mark.asyncio
async def test_commit_audit_index_on_start_commits(monkeypatch):
    settings = SimpleNamespace(
        AUDIT_R2_BUCKET_PUBLIC_URL="https://pub-audit.r2.dev",
        SCOREVISION_PUBLIC_RESULTS_URL="",
        AUDIT_R2_RESULTS_PREFIX="audit_spotcheck",
    )
    monkeypatch.setenv("AUDIT_COMMIT_VALIDATOR_ON_START", "1")
    with patch("scorevision.validator.audit.open_source.storage.get_settings", return_value=settings), \
         patch("scorevision.validator.audit.open_source.storage._commit_audit_index", new=AsyncMock(return_value=True)) as commit_mock:
        await commit_audit_index_on_start()
    assert commit_mock.await_count == 1
    assert (
        commit_mock.await_args.kwargs["index_url"]
        == "https://pub-audit.r2.dev/scorevision/audit_spotcheck/index.json"
    )
