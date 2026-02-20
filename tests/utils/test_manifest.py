import pytest

from scorevision.utils.manifest import (
    Element,
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


def test_manifest_get_element_by_id(dummy_detect_element, dummy_pitch_element):
    man = Manifest(
        window_id="2025-10-27",
        elements=[dummy_detect_element, dummy_pitch_element],
        tee=Tee(trusted_share_gamma=0.2),
        version="1.3",
        expiry_block=123456,
    )
    assert dummy_detect_element == man.get_element(id=dummy_detect_element.id)
    assert dummy_pitch_element == man.get_element(id=dummy_pitch_element.id)


def test_private_track_element_minimal():
    elem = Element(
        id="ActionSpot_v1",
        track="private",
        weight=1.0,
    )
    assert elem.track == "private"
    assert elem.clips == []
    assert elem.metrics is None
    assert elem.preproc is None


def test_private_track_element_in_manifest():
    private_elem = Element(
        id="ActionSpot_v1",
        track="private",
        weight=1.0,
        eval_window=4,
    )
    man = Manifest(
        window_id="2025-12-01",
        elements=[private_elem],
        tee=Tee(trusted_share_gamma=0.0),
        version=1.3,
        expiry_block=999999,
    )
    assert man.get_element("ActionSpot_v1") == private_elem


def test_open_track_element_backward_compatible(dummy_detect_element):
    assert dummy_detect_element.track is None
    assert dummy_detect_element.clips != []
    assert dummy_detect_element.metrics is not None
