import hashlib
import time


def build_signed_headers(keypair, payload_bytes: bytes = b"") -> dict[str, str]:
    nonce = str(int(time.time() * 1e9))
    payload_hash = hashlib.blake2b(payload_bytes, digest_size=32).hexdigest()
    message = f"{nonce}{payload_hash}"
    signature = f"0x{keypair.sign(message.encode('utf-8')).hex()}"

    return {
        "X-Validator-Hotkey": keypair.ss58_address,
        "X-Nonce": nonce,
        "X-Signature": signature,
    }
