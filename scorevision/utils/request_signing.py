import hashlib
import time


def build_signed_headers(
    keypair,
    payload_bytes: bytes = b"",
    miner_hotkey: str | None = None,
) -> dict[str, str]:
    nonce = str(int(time.time() * 1e9))
    payload_hash = hashlib.blake2b(payload_bytes, digest_size=32).hexdigest()
    message = f"{nonce}{payload_hash}"
    signature = f"0x{keypair.sign(message.encode('utf-8')).hex()}"

    headers = {
        "X-Validator-Hotkey": keypair.ss58_address,
        "Validator-Hotkey": keypair.ss58_address,
        "X-Nonce": nonce,
        "Nonce": nonce,
        "X-Signature": signature,
        "Signature": signature,
    }

    if miner_hotkey:
        headers["X-Miner-Hotkey"] = miner_hotkey
        headers["Miner-Hotkey"] = miner_hotkey

    return headers
