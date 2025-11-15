from pathlib import Path

from pytest import fixture
from nacl.signing import SigningKey


@fixture
def keypair():
    """Generate a PyNaCl Ed25519 keypair for signing tests."""
    sk = SigningKey.generate()
    return sk, sk.verify_key


@fixture
def signing_key_hex(keypair) -> str:
    """Return a fresh Ed25519 signing key as hex for env variable injection."""
    key, _ = keypair
    return key.encode().hex()


@fixture
def generated_ed25519_key(tmp_path: Path, signing_key_hex) -> Path:
    """Generate a temporary Ed25519 key as raw 32-byte hex for tests."""

    key_path = tmp_path / "generated_ed25519_key.txt"
    key_path.write_text(signing_key_hex)
    return key_path
