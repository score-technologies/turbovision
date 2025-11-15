from pathlib import Path

import pytest
from nacl.signing import SigningKey
from nacl.encoding import RawEncoder


@pytest.fixture
def signing_key_hex() -> str:
    """Return a fresh Ed25519 signing key as hex for env variable injection."""
    key = SigningKey.generate()
    return key.encode().hex()


@pytest.fixture
def generated_ed25519_key(tmp_path: Path, signing_key_hex) -> Path:
    """Generate a temporary Ed25519 key as raw 32-byte hex for tests."""

    key_path = tmp_path / "generated_ed25519_key.txt"
    key_path.write_text(signing_key_hex)
    return key_path
