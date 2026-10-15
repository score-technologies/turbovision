from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import scorevision.utils.r2 as r2


@pytest.mark.asyncio
async def test_add_index_key_logs_each_stage_for_traced_emission(monkeypatch):
    body = SimpleNamespace(
        read=AsyncMock(return_value=b'["manako/existing.json"]'),
        close=Mock(),
    )
    client = SimpleNamespace(
        head_object=AsyncMock(
            return_value={
                "ContentLength": len(b'["manako/existing.json"]'),
                "ETag": '"etag-1"',
            }
        ),
        get_object=AsyncMock(return_value={"Body": body}),
        put_object=AsyncMock(),
    )

    class ClientContext:
        async def __aenter__(self):
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    logger = Mock()
    monkeypatch.setattr(r2, "logger", logger)
    changed = await r2.add_index_key_if_new(
        client_factory=ClientContext,
        bucket="bucket",
        key="manako/new.json",
        trace_id="trace-1",
    )

    assert changed is True
    messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
    for stage in (
        "index_get",
        "index_read",
        "index_parse",
        "index_serialize",
        "index_put",
    ):
        assert any(
            f"[emit:trace-1] stage={stage} status=start" in message
            for message in messages
        )
        assert any(
            f"[emit:trace-1] stage={stage} status=done" in message
            for message in messages
        )

    put_payload = client.put_object.await_args.kwargs["Body"]
    assert "manako/existing.json" in put_payload
    assert "manako/new.json" in put_payload
    client.get_object.assert_awaited_once_with(
        Bucket="bucket",
        Key="manako/index.json",
        Range="bytes=0-23",
        IfMatch='"etag-1"',
    )


@pytest.mark.asyncio
async def test_read_object_in_ranges_retries_a_stalled_range(monkeypatch):
    monkeypatch.setattr(r2, "_INDEX_RANGE_SIZE", 4)
    monkeypatch.setattr(r2, "_INDEX_RANGE_TIMEOUT_SECONDS", 0.01)

    stalled_body = SimpleNamespace(
        read=AsyncMock(side_effect=TimeoutError),
        close=Mock(),
    )
    first_body = SimpleNamespace(read=AsyncMock(return_value=b"abcd"), close=Mock())
    second_body = SimpleNamespace(read=AsyncMock(return_value=b"ef"), close=Mock())

    client = SimpleNamespace(
        head_object=AsyncMock(
            return_value={"ContentLength": 6, "ETag": '"etag-1"'}
        ),
        get_object=AsyncMock(
            side_effect=[
                {"Body": stalled_body},
                {"Body": first_body},
                {"Body": second_body},
            ]
        ),
    )

    class ClientContext:
        async def __aenter__(self):
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    result = await r2._read_object_in_ranges(
        client_factory=ClientContext,
        bucket="bucket",
        key="manako/index.json",
        trace_id="trace-1",
    )

    assert result == b"abcdef"
    assert client.get_object.await_args_list[0].kwargs["Range"] == "bytes=0-3"
    assert client.get_object.await_args_list[1].kwargs["Range"] == "bytes=0-3"
    assert client.get_object.await_args_list[2].kwargs["Range"] == "bytes=4-5"
    assert all(
        call.kwargs["IfMatch"] == '"etag-1"'
        for call in client.get_object.await_args_list
    )
    stalled_body.close.assert_called_once()
