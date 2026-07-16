from unittest.mock import AsyncMock

import pytest

from scorevision.utils.bittensor_commitments import (
    _decode_commitment_payload,
    get_all_revealed_commitments,
)


def test_decode_commitment_payload_accepts_hex_sdk_format():
    payload = b'{"role":"miner"}'
    encoded = bytes([len(payload) << 2]) + payload
    assert _decode_commitment_payload(encoded.hex()) == payload.decode()


def test_decode_commitment_payload_accepts_asi2_predecoded_format():
    assert _decode_commitment_payload('\x19\x05{"role":"miner"}') == '{"role":"miner"}'


@pytest.mark.asyncio
async def test_commitment_reader_falls_back_to_raw_query_map():
    class Query:
        def __aiter__(self):
            async def values():
                yield "hotkey", [("\x19\x05{\"role\":\"miner\"}", 123)]
            return values()

    subtensor = type("Subtensor", (), {})()
    subtensor.get_all_revealed_commitments = AsyncMock(side_effect=ValueError("not hex"))
    subtensor.query_map = AsyncMock(return_value=Query())

    result = await get_all_revealed_commitments(subtensor, 44)

    assert result == {"hotkey": ((123, '{"role":"miner"}'),)}
