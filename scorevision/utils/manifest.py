from hashlib import sha256
from dataclasses import dataclass, field, asdict
from typing import Any
from json import dumps
from base64 import b64encode, b64decode
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


@dataclass
class Preproc:
    fps: int | None = None
    resize_long: int | None = None
    norm: str | None = None


@dataclass
class Pillars:
    iou: float | None = None
    count: float | None = None
    palette: float | None = None
    smoothness: float | None = None
    role: float | None = None


@dataclass
class Metrics:
    pillars: Pillars | None = None


@dataclass
class Salt:
    offsets: list[int] | None = field(default_factory=list)
    strides: list[int] | None = field(default_factory=list)


@dataclass
class Element:
    id: str
    clips: list[str]
    weights: list[float]
    preproc: Preproc | None = None
    metrics: Metrics | None = None
    latency_p95_ms: int | None = None
    service_rate_fps: int | None = None
    salt: Salt | None = None
    pgt_recipe_hash: str | None = None
    baseline_theta: float | None = None
    delta_floor: float | None = None
    beta: float | None = None


@dataclass
class Tee:
    trusted_share_gamma: float | None = None


@dataclass
class Manifest:
    window_id: str
    elements: list[Element]
    tee: Tee | None = None
    version: str | None = None
    expiry_block: int | None = None
    signature: str | None = None

    def to_canonical_json(self) -> str:
        self.elements.sort(key=lambda element: element.id)
        payload = asdict(self)
        payload.pop("signature", None)
        return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def sign(self, private_key: Ed25519PrivateKey) -> None:
        """Sign the manifest with private key"""
        json_bytes = self.to_canonical_json().encode("utf-8")
        signature = private_key.sign(json_bytes)
        self.signature = b64encode(signature).decode("ascii")

    def verify(self, public_key: Ed25519PublicKey) -> bool:
        try:
            if self.signature is None:
                raise Exception("Manifest is not signed")

            json_bytes = self.to_canonical_json().encode("utf-8")
            signature_bytes = b64decode(self.signature)
            public_key.verify(signature_bytes, json_bytes)
            return True
        except Exception as e:
            print(e)
        return False

    @property
    def hash(self) -> str:
        """Stable SHA-256 hash
        computed over the unsigned manifest payload (signature omitted)
        so that the same manifest content always yields the same hash
        regardless of field order in input dictionaries.
        """
        return sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

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