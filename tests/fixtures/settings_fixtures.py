from types import SimpleNamespace
from json import dumps

from pytest import fixture


@fixture
def fake_settings():
    return SimpleNamespace(
        SCOREVISION_BUCKET="scorevision",
        R2_BUCKET="scorevision",
        R2_BUCKET_PUBLIC_URL="https://pub-test.r2.dev",
        SCOREVISION_ENDPOINT="https://unused",
        SCOREVISION_ACCESS_KEY="x",
        SCOREVISION_SECRET_KEY="y",
        NETWORK="testnet",
        SCOREVISION_M_MIN=1,
        SCOREVISION_WINDOW_TIEBREAK_ENABLE=False,
        SCOREVISION_NETUID=44,
        SCOREVISION_MECHID=1,
        SCOREVISION_WINDOW_DELTA_ABS=0.1,
        SCOREVISION_WINDOW_DELTA_REL=0.1,
        SCOREVISION_VLM_SELECT_N_FRAMES=3,
        BITTENSOR_WALLET_COLD="cold_wallet_name",
        BITTENSOR_WALLET_HOT="hot_wallet_name",
        RUNNER_GET_BLOCK_TIMEOUT_S=15.0,
        RUNNER_WAIT_BLOCK_TIMEOUT_S=15.0,
        RUNNER_RECONNECT_DELAY_S=5.0,
        RUNNER_DEFAULT_ELEMENT_TEMPO=300,
        RUNNER_PGT_MAX_BBOX_RETRIES=3,
        RUNNER_PGT_MAX_QUALITY_RETRIES=4,
        BLOCKS_PER_DAY=7200,
        VALIDATOR_TAIL_BLOCKS=28800,
        VALIDATOR_FALLBACK_UID=6,
        VALIDATOR_WINNERS_EVERY_N=24,
        AUDIT_SPOTCHECK_MIN_INTERVAL_S=7200,
        AUDIT_SPOTCHECK_MAX_INTERVAL_S=14400,
        AUDIT_SPOTCHECK_THRESHOLD=0.95,
    )


@fixture
def fake_index_bytes():
    """Return a fake index JSON encoded as bytes (list of manifest keys)."""
    fake_index = [
        "manifest/123456-abc123.yaml",
        "manifest/123460-def456.yaml",
    ]
    return dumps(fake_index).encode("utf-8")
