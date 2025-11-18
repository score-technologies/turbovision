# tests/fixtures/manifest_fixtures.py

from json import loads
from pathlib import Path

from pytest import fixture
from ruamel.yaml import YAML

from scorevision.utils.manifest import (
    Manifest,
    Tee,
    Element,
    Metrics,
    Preproc,
    Clip,
    PillarName,
)


@fixture
def sample_elements():
    """Provide 3 example Elements for manifest construction."""

    def mk_el(id_, clips):
        return Element(
            id=id_,
            clips=[Clip(hash=c, weight=1.0) for c in clips],
            metrics=Metrics(
                pillars={
                    PillarName.IOU: 0.3,
                    PillarName.COUNT: 0.0,
                    PillarName.PALETTE: 0.7,
                    PillarName.SMOOTHNESS: 0.0,
                    PillarName.ROLE: 0.0,
                }
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
        mk_el("PlayerDetect_v1@1.0", ["b", "a"]),
        mk_el("BallDetect_v1@1.01", ["e", "f"]),
        mk_el("PitchCalib_v1@1.0", ["c", "d"]),
    ]


@fixture
def minimal_manifest(sample_elements):
    """A reusable manifest for tests."""
    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        elements=sample_elements,
        tee=Tee(trusted_share_gamma=0.2),
    )


@fixture
def dummy_manifest():
    """A minimal manifest for publish tests."""
    el = Element(
        id="PlayerDetect_v1",
        clips=[Clip(hash="sha256:abc", weight=1.0)],
        metrics=Metrics(
            pillars={
                PillarName.IOU: 0.3,
                PillarName.COUNT: 0.1,
                PillarName.SMOOTHNESS: 0.3,
                PillarName.ROLE: 0.3,
            }
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


@fixture
def signed_manifest_file(tmp_path: Path, dummy_manifest: Manifest):
    """Write a manifest to YAML (for publish tests)."""
    path = tmp_path / "manifest.yaml"
    raw = loads(dummy_manifest.to_canonical_json())

    yaml = YAML(typ="safe", pure=True)
    with path.open("w") as f:
        yaml.dump(raw, f)

    return path
