# test_manifest.py

import pytest

from scorevision.utils.manifest import (
    Manifest,
    Tee,
)


def test_sign_and_verify_happy_path(minimal_manifest, keypair):
    private_key, public_key = keypair

    minimal_manifest.sign(private_key)
    assert isinstance(minimal_manifest.signature, str)
    assert len(minimal_manifest.signature) > 0

    assert minimal_manifest.verify(public_key)


def test_verify_fails_for_unsigned_manifest(minimal_manifest, keypair):
    _, public_key = keypair

    # Ensure unsigned
    minimal_manifest.signature = None

    with pytest.raises(ValueError):
        minimal_manifest.verify(public_key)


def test_verify_fails_after_tamper(minimal_manifest, keypair):
    private_key, public_key = keypair

    minimal_manifest.sign(private_key)
    minimal_manifest.window_id = "TAMPERED"

    assert minimal_manifest.verify(public_key) is False


def test_manifest_hash_is_stable(sample_elements):
    tee = Tee(trusted_share_gamma=0.2)

    man1 = Manifest(
        window_id="2025-10-27",
        elements=sample_elements,
        tee=tee,
        version="1.3",
        expiry_block=123456,
    )

    # reversed element order
    man2 = Manifest(
        window_id="2025-10-27",
        elements=list(reversed(sample_elements)),
        tee=tee,
        version="1.3",
        expiry_block=123456,
    )

    assert man1.hash == man2.hash

