import asyncio
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from scorevision.utils import predict


class Response:
    def __init__(self, status, body):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.payloads = []

    def post(self, *_args, **_kwargs):
        self.calls += 1
        self.payloads.append(_kwargs["json"])
        response = next(self.responses)
        if response == "hang":
            return HangingResponse()
        return Response(*response)


class HangingResponse:
    async def __aenter__(self):
        await asyncio.sleep(10)

    async def __aexit__(self, *_args):
        return None


@pytest.fixture
def setup_predict(monkeypatch):
    settings = SimpleNamespace(
        CHUTES_MINER_BASE_URL_TEMPLATE="https://example.test/{slug}",
        CHUTES_MINER_PREDICT_ENDPOINT="predict",
        CHUTES_API_KEY=SecretStr("key"),
        SCOREVISION_API_N_RETRIES=0,
        SCOREVISION_BACKOFF_RATE=0,
        SCOREVISION_COLD_START_TIMEOUT_S=0.2,
    )
    monkeypatch.setattr(predict, "get_settings", lambda: settings)
    monkeypatch.setattr(predict, "get_semaphore", lambda: asyncio.Semaphore(1))
    return settings


@pytest.mark.asyncio
async def test_cold_start_retries_original_inference_past_normal_retry_count(
    monkeypatch, setup_predict
):
    session = Session(
        [
            (503, "No instances available"),
            (503, "no instances available"),
            (200, '{"success": true, "predictions": [1]}'),
        ]
    )
    monkeypatch.setattr(predict, "get_async_client", lambda: asyncio.sleep(0, result=session))
    payload = SimpleNamespace(model_dump=lambda **_kwargs: {"input": "original"})

    result = await predict.predict_sv(payload, "miner", chute_id="chute-id")

    assert result.success
    assert result.predictions == [1]
    assert session.calls == 3
    assert session.payloads == [{"input": "original"}] * 3


@pytest.mark.asyncio
async def test_cold_start_retry_is_bounded_even_when_next_request_hangs(
    monkeypatch, setup_predict
):
    session = Session([(503, "No instances available"), "hang"])
    monkeypatch.setattr(predict, "get_async_client", lambda: asyncio.sleep(0, result=session))
    payload = SimpleNamespace(model_dump=lambda **_kwargs: {})

    result = await asyncio.wait_for(
        predict.predict_sv(payload, "miner", chute_id="chute-id"), timeout=1
    )

    assert not result.success
    assert result.error.startswith("chute cold:")
    assert session.calls == 2


@pytest.mark.asyncio
async def test_other_503_does_not_use_cold_start_retry(monkeypatch, setup_predict):
    session = Session([(503, "upstream failure")])
    monkeypatch.setattr(predict, "get_async_client", lambda: asyncio.sleep(0, result=session))
    payload = SimpleNamespace(model_dump=lambda **_kwargs: {})

    result = await predict.predict_sv(payload, "miner", chute_id="chute-id")

    assert not result.success
    assert result.error == "error:RuntimeError:HTTP 503: upstream failure"
    assert session.calls == 1


@pytest.mark.asyncio
async def test_cold_start_budget_cannot_exceed_30_seconds(monkeypatch, setup_predict):
    setup_predict.SCOREVISION_COLD_START_TIMEOUT_S = 120
    setup_predict.SCOREVISION_BACKOFF_RATE = 100
    clock = [0.0]

    async def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(predict, "monotonic", lambda: clock[0])
    monkeypatch.setattr(predict, "sleep", advance)
    session = Session([(503, "No instances available")])
    monkeypatch.setattr(predict, "get_async_client", lambda: asyncio.sleep(0, result=session))
    payload = SimpleNamespace(model_dump=lambda **_kwargs: {})

    result = await predict.predict_sv(payload, "miner")

    assert not result.success
    assert session.calls == 1
    assert clock[0] == 30


@pytest.mark.asyncio
async def test_other_503_uses_normal_retry_count(monkeypatch, setup_predict):
    setup_predict.SCOREVISION_API_N_RETRIES = 3
    session = Session([(503, "upstream failure")] * 4)
    monkeypatch.setattr(predict, "get_async_client", lambda: asyncio.sleep(0, result=session))
    payload = SimpleNamespace(model_dump=lambda **_kwargs: {})

    result = await predict.predict_sv(payload, "miner")

    assert not result.success
    assert session.calls == 4
