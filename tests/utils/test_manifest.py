"""
Pytest for scorevision/utils/manifest.py
"""

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scorevision.utils.manifest import Manifest, Tee, Element


def test_sign_and_verify_happy_path():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    man = Manifest.empty()
    man.sign(private_key=private_key)

    assert isinstance(man.signature, str)
    assert len(man.signature) > 0
    assert man.verify(public_key=public_key)


def test_verify_fails_for_unsigned_manifest():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    man = Manifest.empty()

    assert man.signature is None
    assert not man.verify(public_key=public_key)


def test_verify_fails_after_tamper():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    man = Manifest.empty()
    man.sign(private_key=private_key)
    man.window_id = 999

    assert man.verify(public_key=public_key) is False


def test_manifest_hash_is_stable():
    WINDOW_ID = "2025-10-27"
    VERSION = "1.3"
    TEE = Tee(trusted_share_gamma=0.2)
    EXPIRY_BLOCK = 123456
    ELEMENTS = [
        Element(id=0, clips=["b", "a"], weights=[1.0, 0.0, 0.5]),
        Element(id=1, clips=["e", "f"], weights=[0.9, 0.0, 0.5]),
        Element(id=2, clips=["c", "d"], weights=[0.0, 0.1, 1.0]),
    ]
    ELEMENTS_REVERSED = [
        Element(id=1, clips=["e", "f"], weights=[0.9, 0.0, 0.5]),
        Element(id=2, clips=["c", "d"], weights=[0.0, 0.1, 1.0]),
        Element(id=0, clips=["b", "a"], weights=[1.0, 0.0, 0.5]),
    ]
    man1 = Manifest(
        window_id=WINDOW_ID,
        elements=ELEMENTS,
        tee=TEE,
        version=VERSION,
        expiry_block=EXPIRY_BLOCK,
    )
    man2 = Manifest(
        window_id=WINDOW_ID,
        elements=ELEMENTS_REVERSED,
        tee=TEE,
        version=VERSION,
        expiry_block=EXPIRY_BLOCK,
    )
    assert man1.hash == man2.hash
