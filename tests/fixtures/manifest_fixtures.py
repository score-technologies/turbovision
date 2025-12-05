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
def dummy_objects():
    return [
        "ball",
        "goalkeeper",
        "player",
        "referee",
        "",
        "",
        "team 1",
        "team 2",
    ]


@fixture
def dummy_detect_element(dummy_objects):
    return Element(
        id="PlayerDetect_v1@1.0",
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
        objects=dummy_objects,
    )


@fixture
def dummy_pitch_element():
    return Element(
        id="PitchCalib_v1@1.0",
        clips=[Clip(hash="sha256:abc", weight=1.0)],
        metrics=Metrics(
            pillars={
                PillarName.IOU: 1.0,
            }
        ),
        preproc=Preproc(fps=30, resize_long=720, norm="none"),
        latency_p95_ms=100,
        service_rate_fps=30,
        pgt_recipe_hash="sha256:deadbeef",
        baseline_theta=0.3,
        delta_floor=0.05,
        beta=1.0,
        keypoint_template="football",
    )


@fixture
def dummy_manifest(dummy_detect_element, dummy_pitch_element):
    """A minimal manifest for publish tests."""
    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        elements=[dummy_detect_element, dummy_pitch_element],
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


@fixture
def manifest_with_pillar_that_has_no_metric_registered():
    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        tee=Tee(trusted_share_gamma=0.2),
        elements=[
            Element(
                id="PitchCalib_v1",
                metrics=Metrics(pillars={PillarName.COUNT: 1.0}),
                clips=[Clip(hash="sha256:abc", weight=1.0)],
                preproc=Preproc(fps=30, resize_long=720, norm="none"),
                latency_p95_ms=100,
                service_rate_fps=30,
                pgt_recipe_hash="sha256:deadbeef",
                baseline_theta=0.3,
                delta_floor=0.05,
                beta=1.0,
            )
        ],
    )


@fixture
def manifest_with_pillar_weight_of_zero(dummy_objects):
    return Manifest(
        window_id="2025-10-27",
        version="1.3",
        expiry_block=123456,
        tee=Tee(trusted_share_gamma=0.2),
        elements=[
            Element(
                id="PlayerDetect_v1",
                metrics=Metrics(pillars={PillarName.IOU: 1.0, PillarName.COUNT: 0.0}),
                clips=[Clip(hash="sha256:abc", weight=1.0)],
                preproc=Preproc(fps=30, resize_long=720, norm="none"),
                latency_p95_ms=100,
                service_rate_fps=30,
                pgt_recipe_hash="sha256:deadbeef",
                baseline_theta=0.3,
                delta_floor=0.05,
                beta=1.0,
                objects=dummy_objects,
            )
        ],
    )
