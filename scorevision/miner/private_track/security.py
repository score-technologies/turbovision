import os
from fastapi import Depends, Header, HTTPException, Request
from fiber import constants as cst
from fiber import utils
from fiber.chain import signatures
from fiber.miner.security.nonce_management import NonceManager

BLACKLIST_ENABLED = os.environ.get("BLACKLIST_ENABLED", "true").lower() in ("true", "1", "yes")
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("true", "1", "yes")

_nonce_manager = NonceManager()


async def verify_request(
    request: Request,
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
    signature: str = Header(..., alias=cst.SIGNATURE),
    miner_hotkey: str = Header(..., alias=cst.MINER_HOTKEY),
    nonce: str = Header(..., alias=cst.NONCE),
):
    if not _nonce_manager.nonce_is_valid(nonce):
        raise HTTPException(status_code=401, detail="Invalid nonce")

    body = await request.body()
    payload_hash = signatures.get_hash(body)
    message = utils.construct_header_signing_message(
        nonce=nonce,
        miner_hotkey=miner_hotkey,
        payload_hash=payload_hash,
    )

    if not signatures.verify_signature(
        message=message,
        signer_ss58_address=validator_hotkey,
        signature=signature,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")


def get_security_dependencies() -> list:
    deps = []

    if BLACKLIST_ENABLED:
        from fiber.miner.dependencies import blacklist_low_stake
        deps.append(Depends(blacklist_low_stake))

    if VERIFY_ENABLED:
        deps.append(Depends(verify_request))

    return deps
