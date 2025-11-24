from types import SimpleNamespace
from json import dumps

from pytest import fixture


@fixture
def fake_settings():
    """A fake settings object with all R2/CDN credentials"""
    return SimpleNamespace(
        SCOREVISION_BUCKET="scorevision",
        SCOREVISION_ENDPOINT="https://unused",
        SCOREVISION_ACCESS_KEY="x",
        SCOREVISION_SECRET_KEY="y",
        NETWORK="testnet",
        SCOREVISION_M_MIN=1,
        SCOREVISION_WINDOW_TIEBREAK_ENABLE=False,
        SCOREVISION_NETUID=44,
        SCOREVISION_WINDOW_DELTA_ABS=0.1,
        SCOREVISION_WINDOW_DELTA_REL=0.1,
    )


@fixture
def fake_index_bytes():
    """Return a fake index JSON encoded as bytes."""
    fake_index = {
        "windows": {
            "2025-10-24": {"current": "sha256:abc123", "expiry_block": 123456},
            "2025-10-25": {"current": "sha256:def456", "expiry_block": 123460},
        }
    }
    return dumps(fake_index).encode("utf-8")
