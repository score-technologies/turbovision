# test_manifest.py

import pytest

from scorevision.utils.manifest import (
    Manifest,
    Tee,
)


def test_sign_and_verify_happy_path(dummy_manifest, keypair):
    private_key, public_key = keypair

    dummy_manifest.sign(private_key)
    assert isinstance(dummy_manifest.signature, str)
    assert len(dummy_manifest.signature) > 0

    assert dummy_manifest.verify(public_key)


def test_verify_fails_for_unsigned_manifest(dummy_manifest, keypair):
    _, public_key = keypair

    # Ensure unsigned
    dummy_manifest.signature = None

    with pytest.raises(ValueError):
        dummy_manifest.verify(public_key)


def test_verify_fails_after_tamper(dummy_manifest, keypair):
    private_key, public_key = keypair

    dummy_manifest.sign(private_key)
    dummy_manifest.window_id = "TAMPERED"

    assert dummy_manifest.verify(public_key) is False


def test_manifest_hash_is_stable(dummy_detect_element, dummy_pitch_element):
    tee = Tee(trusted_share_gamma=0.2)

    elements = [dummy_detect_element, dummy_pitch_element]

    man1 = Manifest(
        window_id="2025-10-27",
        elements=elements,
        tee=tee,
        version="1.3",
        expiry_block=123456,
    )

    # reversed element order
    man2 = Manifest(
        window_id="2025-10-27",
        elements=list(reversed(elements)),
        tee=tee,
        version="1.3",
        expiry_block=123456,
    )

    assert man1.hash == man2.hash
