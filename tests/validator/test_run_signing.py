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


def test_signing_hotkey_is_resolved_by_name_like_the_signer(monkeypatch, keypair):
    """Wallet and hotkey names, same resolution as the signer service."""
    from types import SimpleNamespace

    from scorevision.validator.audit.open_source import compliance as compliance_mod

    seen: dict = {}

    def fake_load(wallet_name, hotkey_name):
        seen["wallet"], seen["hotkey"] = wallet_name, hotkey_name
        return keypair

    monkeypatch.setattr(compliance_mod, "load_hotkey_keypair", fake_load)
    monkeypatch.setattr(
        compliance_mod,
        "get_settings",
        lambda: SimpleNamespace(
            CHECKER_SIGNING_WALLET="manako", CHECKER_SIGNING_HOTKEY="latency-h",
            BITTENSOR_WALLET_COLD="default",
        ),
    )
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR", None)
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR_LOADED", False)

    resolved = compliance_mod._signing_keypair()

    assert seen == {"wallet": "manako", "hotkey": "latency-h"}
    assert resolved.ss58_address == keypair.ss58_address


def test_no_hotkey_configured_means_unsigned_runs(monkeypatch):
    from types import SimpleNamespace

    from scorevision.validator.audit.open_source import compliance as compliance_mod

    monkeypatch.setattr(
        compliance_mod,
        "get_settings",
        lambda: SimpleNamespace(CHECKER_SIGNING_WALLET="", CHECKER_SIGNING_HOTKEY=""),
    )
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR", None)
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR_LOADED", False)

    assert compliance_mod._signing_keypair() is None


def test_a_non_sr25519_signing_hotkey_is_refused(monkeypatch):
    """The verifier rebuilds from the ss58, which is sr25519: fail loudly, not silently."""
    from types import SimpleNamespace

    from scorevision.validator.audit.open_source import compliance as compliance_mod

    monkeypatch.setattr(
        compliance_mod,
        "load_hotkey_keypair",
        lambda w, h: SimpleNamespace(crypto_type=0, ss58_address="5Ed25519"),
    )
    monkeypatch.setattr(
        compliance_mod,
        "get_settings",
        lambda: SimpleNamespace(
            CHECKER_SIGNING_WALLET="manako", CHECKER_SIGNING_HOTKEY="latency-h",
            BITTENSOR_WALLET_COLD="default",
        ),
    )
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR", None)
    monkeypatch.setattr(compliance_mod, "_SIGNING_KEYPAIR_LOADED", False)

    with pytest.raises(ValueError, match="signing_hotkey_not_sr25519"):
        compliance_mod._signing_keypair()
