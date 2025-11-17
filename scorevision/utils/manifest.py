"""
Manifest data structures and signing utilities.

This module defines the canonical schema for a Score Vision Manifest.
A Manifest is a cryptographically signed, content-addressed rulebook
for a single evaluation window. It specifies Elements, metrics, baselines,
latency gates, service rates, and TEE-related trust parameters.
"""

from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass, field, asdict
from json import dumps
from base64 import b64encode, b64decode
from enum import Enum
from functools import cached_property
from json import loads

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
from ruamel.yaml import YAML


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


@dataclass
class Preproc:
    """
    Preprocessing parameters applied before evaluation.
    """

    fps: int
    resize_long: int
    norm: NormType


@dataclass
class Pillars:
    """
    Multi-pillar scoring weights for an Element.
    Keys must match PillarName enums.
    """

    iou: float
    count: float
    palette: float
    smoothness: float
    role: float


@dataclass
class Metrics:
    """
    Metrics configuration used to score Element performance.
    """

    pillars: Pillars


@dataclass
class Salt:
    """
    VRF-derived challenge salting parameters ensuring per-validator
    unpredictability. These are always present (default empty lists).
    """

    offsets: list[int] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)


@dataclass
class Clip:
    """
    Clip definition used in YAML configs.
    Structured as:
      - hash: "sha256:..."
        weight: 1.0
    """

    hash: str
    weight: float


@dataclass
class Element:
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
    salt: Salt = field(default_factory=Salt)


@dataclass
class Tee:
    """
    Trusted Execution Environment parameters.
    Defines how much reward share comes from Trusted Track.
    """

    trusted_share_gamma: float


@dataclass
class Manifest:
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
    version: str
    expiry_block: int
    elements: list[Element]
    tee: Tee
    signature: str | None = None

    @classmethod
    def load_yaml(cls, path: Path) -> "Manifest":
        data = yaml.load(path.read_text())
        elements = []
        for e in data["elements"]:
            pillars = Pillars(**e["metrics"]["pillars"])
            metrics = Metrics(pillars=pillars)
            clips = [Clip(hash=c["hash"], weight=c["weight"]) for c in e["clips"]]
            preproc = Preproc(
                fps=e["preproc"]["fps"],
                resize_long=e["preproc"]["resize_long"],
                norm=e["preproc"]["norm"],
            )
            element = Element(
                id=e["id"],
                clips=clips,
                metrics=metrics,
                preproc=preproc,
                latency_p95_ms=e["latency_p95_ms"],
                service_rate_fps=e["service_rate_fps"],
                pgt_recipe_hash=e["pgt_recipe_hash"],
                baseline_theta=e["baseline_theta"],
                delta_floor=e["delta_floor"],
                beta=e["beta"],
                salt=Salt(
                    offsets=e.get("salt", {}).get("offsets", []),
                    strides=e.get("salt", {}).get("strides", []),
                ),
            )
            elements.append(element)
        tee = Tee(trusted_share_gamma=data["tee"]["trusted_share_gamma"])
        return Manifest(
            window_id=data["window_id"],
            version=data["version"],
            expiry_block=data["expiry_block"],
            elements=elements,
            tee=tee,
            signature=data.get("signature"),
        )

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
        payload = asdict(self)
        elements_sorted = sorted(self.elements, key=lambda e: e.id)
        payload["elements"] = [asdict(e) for e in elements_sorted]
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


if __name__ == "__main__":
    man = Manifest.load_yaml(path=Path("scorevision/example.yml"))
    print(man)
