from pathlib import Path
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519


@pytest.fixture
def generated_pem_key(tmp_path: Path) -> Path:
    """Generate EC private key."""
    key = ec.generate_private_key(ec.SECP256R1())
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_path = tmp_path / "generated_ec_key.pem"
    pem_path.write_bytes(pem_bytes)
    return pem_path


@pytest.fixture
def generated_ed25519_key(tmp_path: Path) -> Path:
    """Generate a temporary Ed25519 PEM key for tests."""
    private_key = ed25519.Ed25519PrivateKey.generate()

    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,  # PKCS8 is required by many tools
        encryption_algorithm=serialization.NoEncryption(),
    )

    pem_path = tmp_path / "generated_ed25519_key.pem"
    pem_path.write_bytes(pem_bytes)
    return pem_path

