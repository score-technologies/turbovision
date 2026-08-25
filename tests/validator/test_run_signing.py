import pytest
from bittensor_wallet import Keypair

from scorevision.utils.run_signing import (
    canonical_bytes,
    sign_run_payload,
    verify_run_payload,
)


RUN_KEY = "compliance/runs/008908950.json"
TUPLE = ("hk1", "manak0/Detect-fire", 123)


def _payload(status: str = "PENDING_LATENCY", p95: float = 115.0) -> dict:
    return {
        "type": "public_compliance_run",
        "ts": 1.0,
        "winners_block": 8908950,
        "targets": 1,
        "results": [
            {
                "hotkey": TUPLE[0],
                "element_id": TUPLE[1],
                "commit_block": TUPLE[2],
                "status": status,
                "p95_latency_ms": p95,
                "effective_latency_threshold_ms": 110.0,
            }
        ],
    }


@pytest.fixture
def keypair() -> Keypair:
    return Keypair.create_from_uri("//RunSigner")


def test_canonical_bytes_ignore_key_order_and_signature_fields(keypair):
    signed = sign_run_payload(_payload(), run_key=RUN_KEY, keypair=keypair)
    reordered = dict(reversed(list(signed.items())))

    assert canonical_bytes(signed) == canonical_bytes(reordered)


def test_signature_is_bound_to_content_and_object_key(keypair):
    signed = sign_run_payload(_payload(), run_key=RUN_KEY, keypair=keypair)
    hotkey = keypair.ss58_address

    assert verify_run_payload(signed, run_key=RUN_KEY, expected_hotkey=hotkey)
    # copied onto another object key
    assert not verify_run_payload(signed, run_key="compliance/runs/000000001.json", expected_hotkey=hotkey)
    # content altered after signing
    tampered = dict(signed)
    tampered["results"] = _payload(status="FAIL_LATENCY")["results"]
    assert not verify_run_payload(tampered, run_key=RUN_KEY, expected_hotkey=hotkey)
    # signed by someone else
    other = Keypair.create_from_uri("//Attacker")
    forged = sign_run_payload(_payload(), run_key=RUN_KEY, keypair=other)
    assert not verify_run_payload(forged, run_key=RUN_KEY, expected_hotkey=hotkey)
