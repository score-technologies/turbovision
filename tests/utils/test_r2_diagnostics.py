from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import scorevision.utils.r2 as r2


@pytest.mark.asyncio
async def test_add_index_key_logs_each_stage_for_traced_emission(monkeypatch):
    body = AsyncMock()
    body.read.return_value = b'["manako/existing.json"]'
    client = SimpleNamespace(
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
