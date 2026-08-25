from __future__ import annotations

from json import dumps
from logging import getLogger
from pathlib import Path
from typing import Any

from bittensor_wallet import Keypair

logger = getLogger(__name__)

SIGNATURE_FIELD = "signature"
SIGNER_FIELD = "signer_hotkey"
RUN_KEY_FIELD = "run_key"


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a run payload deterministically, signature fields excluded.

    Both sides must produce the exact same bytes, so key order and separators are
    pinned here rather than left to the caller.
    """
    body = {k: v for k, v in payload.items() if k not in (SIGNATURE_FIELD, SIGNER_FIELD)}
    return dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_signing_keypair(key_file: str | Path) -> Keypair:
    """Load the run-signing key from a file the miner worker cannot read (root:0600).

    Accepts a hex seed or a mnemonic; the file never travels through the
    environment, which is scrubbed before any miner code runs.
    """
    raw = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty_signing_key_file:{key_file}")
    secret = raw.splitlines()[0].strip()
    if secret.startswith("0x") and len(secret) == 66:
        return Keypair.create_from_seed(secret)
    return Keypair.create_from_mnemonic(secret)


def sign_run_payload(payload: dict[str, Any], *, run_key: str, keypair: Keypair) -> dict[str, Any]:
    """Bind the payload to its object key, then sign it.

    Without the key inside the signed bytes, an authentic run could be copied to
    another object key and still verify.
    """
    signed = dict(payload)
    signed[RUN_KEY_FIELD] = str(run_key)
    signature = keypair.sign(canonical_bytes(signed))
    signed[SIGNATURE_FIELD] = f"0x{signature.hex()}"
    signed[SIGNER_FIELD] = keypair.ss58_address
    return signed


def verify_run_payload(payload: Any, *, run_key: str, expected_hotkey: str) -> bool:
    if not isinstance(payload, dict):
        return False

    signature = str(payload.get(SIGNATURE_FIELD) or "").strip()
    signer = str(payload.get(SIGNER_FIELD) or "").strip()
    expected = str(expected_hotkey or "").strip()
    if not signature or not signer or not expected or signer != expected:
        return False
    if str(payload.get(RUN_KEY_FIELD) or "").strip() != str(run_key).strip():
        return False

    try:
        raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
        return bool(Keypair(ss58_address=signer).verify(canonical_bytes(payload), raw))
    except Exception as e:
        logger.warning("[run-signing] verification failed run_key=%s err=%s", run_key, e)
        return False
