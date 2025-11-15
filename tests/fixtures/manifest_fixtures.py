# tests/fixtures/manifest_fixtures.py

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from ruamel.yaml import YAML

from scorevision.utils.manifest import (
    Manifest,
    Tee,
    Element,
    Metrics,
    Pillars,
    Preproc,
    Clip,
)

# ------------------------------------------------------------
# SETTINGS FIXTURE
# ------------------------------------------------------------


@pytest.fixture
def fake_settings():
    """A fake settings object with all R2/CDN credentials"""
    return SimpleNamespace(
        SCOREVISION_BUCKET="scorevision",
        SCOREVISION_ENDPOINT="https://unused",
        SCOREVISION_ACCESS_KEY="x",
        SCOREVISION_SECRET_KEY="y",
        NETWORK="testnet",
    )


# ------------------------------------------------------------
# KEYPAIR FIXTURE
# ------------------------------------------------------------


@pytest.fixture
def keypair():
    """Generate a PyNaCl Ed25519 keypair for signing tests."""
    sk = SigningKey.generate()
    return sk, sk.verify_key


# ------------------------------------------------------------
# ELEMENT FIXTURES
# ------------------------------------------------------------


@pytest.fixture
def sample_elements():
    """Provide 3 example Elements for manifest construction."""

    def mk_el(id_, clips):
        return Element(
            id=id_,
            clips=[Clip(hash=c, weight=1.0) for c in clips],
            metrics=Metrics(
                pillars=Pillars(
                    iou=1.0, count=0.0, palette=0.5, smoothness=0.0, role=0.0
                )
            ),
            preproc=Preproc(fps=30, resize_long=720, norm="none"),
            latency_p95_ms=100,
            service_rate_fps=30,
            pgt_recipe_hash="sha256:deadbeef",
            baseline_theta=0.3,
            delta_floor=0.05,
            beta=1.0,
        )

    return [
        mk_el("0", ["b", "a"]),
        mk_el("1", ["e", "f"]),
        mk_el("2", ["c", "d"]),
    ]


# ------------------------------------------------------------
# MANIFEST FIXTURES
# ------------------------------------------------------------


@pytest.fixture
def minimal_manifest(sample_elements):
    """A reusable manifest for tests."""
    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        elements=sample_elements,
        tee=Tee(trusted_share_gamma=0.2),
    )


@pytest.fixture
def dummy_manifest():
    """A minimal manifest for publish tests."""
    el = Element(
        id="TestElement",
        clips=[Clip(hash="sha256:abc", weight=1.0)],
        metrics=Metrics(
            pillars=Pillars(iou=1.0, count=0.0, palette=0.5, smoothness=0.0, role=0.0)
        ),
        preproc=Preproc(fps=30, resize_long=720, norm="none"),
        latency_p95_ms=100,
        service_rate_fps=30,
        pgt_recipe_hash="sha256:deadbeef",
        baseline_theta=0.3,
        delta_floor=0.05,
        beta=1.0,
    )

    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        elements=[el],
        tee=Tee(trusted_share_gamma=0.2),
    )


# ------------------------------------------------------------
# SIGNED MANIFEST FILE FIXTURE
# ------------------------------------------------------------


@pytest.fixture
def signed_manifest_file(tmp_path: Path, dummy_manifest: Manifest):
    """Write a manifest to YAML (for publish tests)."""
    path = tmp_path / "manifest.yaml"
    raw = json.loads(dummy_manifest.to_canonical_json())

    yaml = YAML(typ="safe", pure=True)
    with path.open("w") as f:
        yaml.dump(raw, f)

    return path
