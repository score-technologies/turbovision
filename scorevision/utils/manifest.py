from hashlib import sha256
from typing import Any
from json import dumps
from base64 import b64encode, b64decode

from pydantic import BaseModel
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class Preproc(BaseModel):
    fps: int | None = None
    resize_long: int | None = None
    norm: str | None = None


class Pillars(BaseModel):
    iou: float | None = None
    count: float | None = None
    palette: float | None = None
    smoothness: float | None = None
    role: float | None = None


class Metrics(BaseModel):
    pillars: Pillars | None = None


class Salt(BaseModel):
    offsets: list[int] | None = None
    strides: list[int] | None = None


class Element(BaseModel):
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


class Tee(BaseModel):
    trusted_share_gamma: float | None = None


class Manifest(BaseModel):
    window_id: str
    elements: list[Element]
    tee: Tee
    version: str
    expiry_block: int 
    signature: str | None = None

    @classmethod
    def empty(cls) -> "Manifest":
        return cls(
            window_id="template",
            elements=[
               Element(
                  id="test",
                  clips=[],
                  weights=[] 
               )
            ],
            tee=Tee(),
            version="1.0",
            expiry_block=0,        
        )

    def to_canonical_json(self) -> str:
        sorted_elements = sorted(self.elements, key=lambda e: e.id)
        payload = self.model_dump(exclude={"signature"}, by_alias=True)
        payload["elements"] = [e.model_dump() for e in sorted_elements]
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
