# tests/fixtures/manifest_fixtures.py

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey
)

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
# KEYPAIR FIXTURE
# ------------------------------------------------------------

@pytest.fixture
def keypair():
    """Generate an Ed25519 keypair for signing tests."""
    private = Ed25519PrivateKey.generate()
    return private, private.public_key()


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
                    iou=1.0, count=0.0, palette=0.5,
                    smoothness=0.0, role=0.0
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
# MANIFEST FIXTURE
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

