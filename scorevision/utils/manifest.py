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
from enum import Enum
from functools import cached_property
from json import loads

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
from ruamel.yaml import YAML
from pydantic import BaseModel, Field


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

    def save_yaml(self, path: Path) -> None:
        raw = loads(self.to_canonical_json())
        if self.signature:
            raw["signature"] = self.signature
        yaml.dump(raw, path.open("w"))
