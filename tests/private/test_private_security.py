import time

import pytest
from fastapi import HTTPException
from fiber import utils
from fiber.chain import signatures
from bittensor_wallet import Keypair
from starlette.requests import Request

from scorevision.miner.private_track.security import verify_request

TARGET_MINER_HOTKEY = "5DFwFpurRFaT5VtjdsATUnRakGhvUdgCghF61457QbWdsnJp"


def _make_request(body: bytes) -> Request:
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/challenge",
        "headers": [],
    }
    return Request(scope, receive)


def _make_nonce() -> str:
    return f"{time.time_ns()}_nonce"


@pytest.mark.asyncio
async def test_verify_request_accepts_valid_fiber_signature_for_target_miner_hotkey():
    validator_keypair = Keypair.create_from_uri("//Alice")
    body = b'{"challenge_id":"c1","video_url":"https://example.com/v.mp4"}'
    request = _make_request(body)

    nonce = _make_nonce()
    payload_hash = signatures.get_hash(body)
    message = utils.construct_header_signing_message(
        nonce=nonce,
        miner_hotkey=TARGET_MINER_HOTKEY,
        payload_hash=payload_hash,
    )
    signature = signatures.sign_message(validator_keypair, message)

    await verify_request(
        request=request,
        validator_hotkey=validator_keypair.ss58_address,
        signature=signature,
        miner_hotkey=TARGET_MINER_HOTKEY,
        nonce=nonce,
    )


@pytest.mark.asyncio
async def test_verify_request_rejects_signature_if_miner_hotkey_header_changes():
    validator_keypair = Keypair.create_from_uri("//Alice")
    body = b'{"challenge_id":"c1","video_url":"https://example.com/v.mp4"}'
    request = _make_request(body)

    nonce = _make_nonce()
    payload_hash = signatures.get_hash(body)
    message = utils.construct_header_signing_message(
        nonce=nonce,
        miner_hotkey=TARGET_MINER_HOTKEY,
        payload_hash=payload_hash,
    )
    signature = signatures.sign_message(validator_keypair, message)

    with pytest.raises(HTTPException, match="Invalid signature"):
        await verify_request(
            request=request,
            validator_hotkey=validator_keypair.ss58_address,
            signature=signature,
            miner_hotkey="5FakeDifferentMinerHotkey11111111111111111111111111",
            nonce=nonce,
        )


@pytest.mark.asyncio
async def test_verify_request_rejects_when_validator_hotkey_is_spoofed():
    validator_keypair = Keypair.create_from_uri("//Alice")
    body = b'{"challenge_id":"c1","video_url":"https://example.com/v.mp4"}'
    request = _make_request(body)

    nonce = _make_nonce()
    payload_hash = signatures.get_hash(body)
    message = utils.construct_header_signing_message(
        nonce=nonce,
        miner_hotkey=TARGET_MINER_HOTKEY,
        payload_hash=payload_hash,
    )
    signature = signatures.sign_message(validator_keypair, message)

    with pytest.raises(HTTPException, match="Invalid signature"):
        await verify_request(
            request=request,
            validator_hotkey=TARGET_MINER_HOTKEY,
            signature=signature,
            miner_hotkey=TARGET_MINER_HOTKEY,
            nonce=nonce,
        )
