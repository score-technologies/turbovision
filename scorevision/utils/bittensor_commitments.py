"""Compatibility helpers for Bittensor commitment reads."""

from string import hexdigits


def _scale_offset(first_byte: int) -> int:
    mode = first_byte & 0b11
    if mode == 0:
        return 1
    if mode == 1:
        return 2
    return 4


def _decode_commitment_payload(payload) -> str:
    """Decode both SDK hex payloads and ASI 2/cyscale pre-decoded strings."""
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        return raw[_scale_offset(raw[0]) :].decode("utf-8", errors="ignore") if raw else ""

    text = str(payload or "")
    if not text:
        return ""

    candidate = text.removeprefix("0x")
    if candidate and len(candidate) % 2 == 0 and all(ch in hexdigits for ch in candidate):
        raw = bytes.fromhex(candidate)
        return raw[_scale_offset(raw[0]) :].decode("utf-8", errors="ignore") if raw else ""

    if text[0] in "[{":
        return text

    return text[_scale_offset(ord(text[0])) :]


async def get_all_revealed_commitments(subtensor, netuid: int):
    """Read commitments across SDK hex and ASI 2 pre-decoded return formats."""
    try:
        return await subtensor.get_all_revealed_commitments(netuid)
    except ValueError:
        query = await subtensor.query_map(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid],
        )
        result = {}
        async for hotkey, values in query:
            result[str(hotkey)] = tuple(
                (int(block), _decode_commitment_payload(payload))
                for payload, block in values
            )
        return result
