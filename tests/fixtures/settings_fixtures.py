from pytest import fixture
from types import SimpleNamespace


@fixture
def fake_settings():
    """A fake settings object with all R2/CDN credentials"""
    return SimpleNamespace(
        SCOREVISION_BUCKET="scorevision",
        SCOREVISION_ENDPOINT="https://unused",
        SCOREVISION_ACCESS_KEY="x",
        SCOREVISION_SECRET_KEY="y",
        NETWORK="testnet",
    )
