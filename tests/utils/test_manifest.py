import pytest

from scorevision.utils.manifest import (
    Element,
    Manifest,
    Tee,
    Element,
    Metrics,
    Preproc,
    PillarName,
    _pick_manifest_url_for_block,
    _join_key_to_base,
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


def test_pick_manifest_url_for_block_uses_latest_when_no_block():
    urls = [
        "https://example.com/manifest/100-a.yaml",
        "https://example.com/manifest/250-b.yaml",
        "https://example.com/manifest/180-c.yaml",
    ]
    picked = _pick_manifest_url_for_block(urls, None)
    assert picked == (250, "https://example.com/manifest/250-b.yaml")


def test_pick_manifest_url_for_block_uses_latest_eligible():
    urls = [
        "https://example.com/manifest/100-a.yaml",
        "https://example.com/manifest/180-c.yaml",
        "https://example.com/manifest/250-b.yaml",
    ]
    picked = _pick_manifest_url_for_block(urls, 200)
    assert picked == (180, "https://example.com/manifest/180-c.yaml")


def test_join_key_to_base_normalizes_dot_slash_manifest_key():
    index_url = "https://turbo.scoredata.me/manifest/index.json"
    key = "./manifest/7885821-abc.yaml"
    assert _join_key_to_base(index_url, key) == "https://turbo.scoredata.me/manifest/7885821-abc.yaml"


def test_join_key_to_base_strips_spaces_around_manifest_key():
    index_url = "https://turbo.scoredata.me/manifest/index.json"
    key = " manifest/7885821-abc.yaml "
    assert _join_key_to_base(index_url, key) == "https://turbo.scoredata.me/manifest/7885821-abc.yaml"


def test_element_allows_missing_clips_and_pgt_recipe_hash():
    man = Manifest(
        window_id="2025-10-27",
        elements=[
            Element(
                id="Detect_v1@1.0",
                metrics=Metrics(pillars={PillarName.IOU: 1.0}),
                preproc=Preproc(fps=5, resize_long=1280, norm="rgb-01"),
                latency_p95_ms=200,
                service_rate_fps=25,
                baseline_theta=0.0,
                delta_floor=0.01,
                beta=1.0,
            )
        ],
        tee=Tee(trusted_share_gamma=0.2),
        version="1.3",
        expiry_block=123456,
    )

    elem = man.elements[0]
    assert elem.clips == []
    assert elem.pgt_recipe_hash is None


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
