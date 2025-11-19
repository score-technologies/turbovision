"""
Manifest data structures and signing utilities.

This module defines the canonical schema for a Score Vision Manifest.
A Manifest is a cryptographically signed, content-addressed rulebook
for a single evaluation window. It specifies Elements, metrics, baselines,
latency gates, service rates, and TEE-related trust parameters.
"""

from pathlib import Path
from hashlib import sha256
from json import dumps
from base64 import b64encode, b64decode
import json
import os
from pathlib import Path
from enum import Enum
from functools import cached_property
from json import loads

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
from ruamel.yaml import YAML
from pydantic import BaseModel, Field, model_validator


yaml = YAML()
yaml.default_flow_style = False

# ------------------------------------------------------------
# ENUMS
# ------------------------------------------------------------


class NormType(str, Enum):
    """Preprocessing normalisation modes."""

    RGB_01 = "rgb-01"
    RGB_255 = "rgb-255"
    NONE = "none"


class PillarName(str, Enum):
    """
    Multi-pillar metrics used by Elements.
    (See Appendix E for naming conventions).
    """

    IOU = "iou"
    COUNT = "count"
    PALETTE = "palette"
    SMOOTHNESS = "smoothness"
    ROLE = "role"


class ElementPrefix(str, Enum):
    """Prefix to Element Names
    e.g.
    - PlayerDetect_v1: Object detection and tracking at. Outputs include bounding boxes, tracking IDs, team assignments, and role classifications.
    - BallDetect_v1: Small object tracking. Handles fast-moving objects with motion blur and frequent occlusions.
    - PitchCalib_v1: Geometric calibration. Outputs keypoint locations and homography matrices for image-to-field coordinate transformation.
    """

    PLAYER_DETECTION = "PlayerDetect"
    BALL_DETECTION = "BallDetect"
    PITCH_CALIBRATION = "PitchCalib"


# ------------------------------------------------------------
# DATA CLASSES
# ------------------------------------------------------------


class Preproc(BaseModel):
    """
    Preprocessing parameters applied before evaluation.
    """

    fps: int
    resize_long: int
    norm: NormType


class Metrics(BaseModel):
    """
    Metrics configuration used to score Element performance.
    """

    pillars: dict[PillarName, float]

    @model_validator(mode="after")
    def validate_pillar_weights(self):
        """Ensure the Weights for all pillars sum to 1.0"""
        if sum(self.pillars.values()) != 1.0:
            raise ValueError(f"Weights must sum to 1.0: {self.pillars.values()}")
        return self


class Salt(BaseModel):
    """
    VRF-derived challenge salting parameters ensuring per-validator
    unpredictability. These are always present (default empty lists).
    """

    offsets: list[int] = Field(default_factory=list)
    strides: list[int] = Field(default_factory=list)


class Clip(BaseModel):
    """
    Clip definition used in YAML configs.
    Structured as:
      - hash: "sha256:..."
        weight: 1.0
    """

    hash: str
    weight: float


class Element(BaseModel):
    """
    Atomic capability definition (e.g., PlayerDetect, BallDetect).

    Fields:
      id:                Unique versioned name (e.g., "PlayerDetect_v1@1.0")
      clips:             List of Clip objects
      preproc:           Preprocessing config
      metrics:           Metric pillar weights
      latency_p95_ms:    Hard p95 latency gate
      service_rate_fps:  Target real-time service rate
      pgt_recipe_hash:   Immutable PGT recipe hash
      baseline_theta:    Score threshold for emissions
      delta_floor:       Minimum margin above baseline
      beta:              Difficulty weight
    """

    id: str
    clips: list[Clip]
    metrics: Metrics
    preproc: Preproc
    latency_p95_ms: int
    service_rate_fps: int
    pgt_recipe_hash: str
    baseline_theta: float
    delta_floor: float
    beta: float
    salt: Salt = Field(default_factory=Salt)

    @property
    def category(self) -> ElementPrefix:
        for element_prefix in ElementPrefix:
            if self.id.startswith(element_prefix):
                return element_prefix
        raise ValueError(f"Unrecognised element {self.id}")

    @model_validator(mode="after")
    def validate_id(self):
        self.category
        return self


class Tee(BaseModel):
    """
    Trusted Execution Environment parameters.
    Defines how much reward share comes from Trusted Track.
    """

    trusted_share_gamma: float


class Manifest(BaseModel):
    """
    Canonical Manifest representing one evaluation window.

    Fields:
      window_id:     Unique ID for the window (YYYY-MM-DD)
      elements:      List of Elements with full scoring/configuration
      tee:           TEE trust parameters
      version:       Manifest schema version (e.g., "1.3")
      expiry_block:  Block height after which the manifest expires
      signature:     Base64-encoded Ed25519 signature (optional)

    Methods:
      to_canonical_json() → Stable JSON for hashing/signing.
      sign(private_key)   → Apply Ed25519 signature.
      verify(public_key)  → Verify Ed25519 signature.
      hash                → SHA-256 content-address.
    """

    window_id: str
    version: float
    expiry_block: int
    elements: list[Element]
    tee: Tee
    signature: str | None = None

    @classmethod
    def load_yaml(cls, path: Path) -> "Manifest":
        data = yaml.load(path.read_text())
        return Manifest(**data)

    def to_canonical_json(self) -> str:
        """
        Produce deterministic canonical JSON suitable for hashing and signing.
        The canonical JSON representation is stable
        and signature-independent
        (i.e. the signature is excluded from hashing and signing)
        so that content hashing is deterministic.
        - Sort Elements lexicographically by ID
        - Exclude signature
        - Compact separators
        - Sorted keys
        """
        payload = self.model_dump(mode="json")
        elements_sorted = sorted(self.elements, key=lambda e: e.id)
        payload["elements"] = [e.model_dump(mode="json") for e in elements_sorted]
        payload.pop("signature", None)
        return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def sign(self, signing_key: SigningKey) -> None:
        """
        Sign the canonical manifest using an Ed25519 private key.
        """
        self.signature = signing_key.sign(
            self.to_canonical_json().encode("utf-8")
        ).signature.hex()

    def verify(self, verify_key: VerifyKey) -> bool:
        """
        Verify the manifest's signature with the corresponding public key.
        """
        if not self.signature:
            raise ValueError("Manifest has no signature to verify.")
        try:
            verify_key.verify(
                self.to_canonical_json().encode("utf-8"), bytes.fromhex(self.signature)
            )
            return True
        except BadSignatureError:
            return False
        except Exception as e:
            raise ValueError(f"Signature verification failed: {e}")

    @cached_property
    def hash(self) -> str:
        """
        Deterministic hashing by
        computing a stable SHA-256 hash of the manifest content
        (This makes the Manifest content-addressable)
        """
        return sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

<<<<<<< HEAD
    @property
    def manifest_hash(self) -> str:
        """Alias used by the protocol text."""
        return self.hash

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        """Rebuild a Manifest (and nested dataclasses) from a plain dict (e.g. JSON)."""
        elements: list[Element] = []
        for e in data.get("elements", []):
            preproc = None
            if e.get("preproc"):
                preproc = Preproc(**e["preproc"])

            metrics = None
            if e.get("metrics"):
                pillars = None
                m = e["metrics"]
                if m.get("pillars"):
                    pillars = Pillars(**m["pillars"])
                metrics = Metrics(pillars=pillars)

            salt = None
            if e.get("salt"):
                salt = Salt(**e["salt"])

            elements.append(
                Element(
                    id=e["id"],
                    clips=e.get("clips", []),
                    weights=e.get("weights", []),
                    preproc=preproc,
                    metrics=metrics,
                    latency_p95_ms=e.get("latency_p95_ms"),
                    service_rate_fps=e.get("service_rate_fps"),
                    salt=salt,
                    pgt_recipe_hash=e.get("pgt_recipe_hash"),
                    baseline_theta=e.get("baseline_theta"),
                    delta_floor=e.get("delta_floor"),
                    beta=e.get("beta"),
                )
            )

        tee = None
        if data.get("tee"):
            tee = Tee(**data["tee"])

        return cls(
            window_id=data["window_id"],
            elements=elements,
            tee=tee,
            version=data.get("version"),
            expiry_block=data.get("expiry_block"),
            signature=data.get("signature"),
        )

def load_manifest_from_file(path: str | Path) -> Manifest:
    """Load a Manifest from a JSON file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest file not found at: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest.from_dict(data)


def get_current_manifest(block_number: int | None = None) -> Manifest:
    """
    Return the current active Manifest.

    Phase 2 version:
    - reads JSON path from env SCOREVISION_MANIFEST_PATH or SV_MANIFEST_PATH
    - optionally checks expiry_block against block_number if provided
    """
    path = os.getenv("SCOREVISION_MANIFEST_PATH") or os.getenv("SV_MANIFEST_PATH")
    if not path:
        raise RuntimeError(
            "No manifest path configured. Set SCOREVISION_MANIFEST_PATH or SV_MANIFEST_PATH."
        )

    manifest = load_manifest_from_file(path)

    if (
        block_number is not None
        and manifest.expiry_block is not None
        and block_number > manifest.expiry_block
    ):
        raise RuntimeError(
            f"Manifest expired at block {manifest.expiry_block}, current block={block_number}."
        )

    return manifest
=======
    def save_yaml(self, path: Path) -> None:
        raw = loads(self.to_canonical_json())
        if self.signature:
            raw["signature"] = self.signature
        yaml.dump(raw, path.open("w"))
>>>>>>> toward-manako
