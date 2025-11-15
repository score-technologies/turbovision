from pathlib import Path
import pytest
from nacl.signing import SigningKey
from nacl.encoding import RawEncoder


@pytest.fixture
def signing_key_hex():
    """Return a fresh Ed25519 signing key as hex for env variable injection."""
    key = SigningKey.generate()
    return key.encode().hex()


@pytest.fixture
def generated_ed25519_key(tmp_path: Path) -> Path:
    """Generate a temporary Ed25519 key as raw 32-byte hex for tests."""
    signing_key = SigningKey.generate()
    raw_hex = signing_key.encode().hex()  # 32-byte seed → hex

    key_path = tmp_path / "generated_ed25519_key.txt"
    key_path.write_text(raw_hex)
    return key_path
