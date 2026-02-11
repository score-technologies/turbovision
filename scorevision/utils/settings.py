from os import getenv
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

__version__ = "0.2.0"


class Settings(BaseModel):
    # Bittensor
    BITTENSOR_WALLET_COLD: str
    BITTENSOR_WALLET_HOT: str
    BITTENSOR_WALLET_PATH: Path
    BITTENSOR_SUBTENSOR_ENDPOINT: str
    BITTENSOR_SUBTENSOR_FALLBACK: str

    # Chutes
    CHUTES_USERNAME: str
    CHUTES_VLM: str
    CHUTES_VLM_ENDPOINT: str
    CHUTES_SAM3_ENDPOINT: str
    CHUTES_SAM3_ID: str
    CHUTES_MINERS_ENDPOINT: str
    CHUTES_MINER_PREDICT_ENDPOINT: str
    CHUTES_MINER_BASE_URL_TEMPLATE: str
    CHUTES_API_KEY: SecretStr
    PATH_CHUTE_TEMPLATES: Path
    PATH_CHUTE_SCRIPT: Path
    FILENAME_CHUTE_MAIN: str
    FILENAME_CHUTE_SCHEMAS: str
    FILENAME_CHUTE_SETUP_UTILS: str
    FILENAME_CHUTE_LOAD_UTILS: str
    FILENAME_CHUTE_PREDICT_UTILS: str

    # OpenRouter
    OPENROUTER_API_KEY: SecretStr
    OPENROUTER_VLM_ENDPOINT: str
    OPENROUTER_VLM: str

    # HuggingFace
    HUGGINGFACE_USERNAME: str
    HUGGINGFACE_API_KEY: SecretStr
    HUGGINGFACE_CONCURRENCY: int

    # Central and Audit Validator 
    URL_MANIFEST: str
    SCOREVISION_BUCKET: str
    SCOREVISION_PUBLIC_RESULTS_URL: str

    # Central Validator Cloudflare R2
    CENTRAL_R2_ACCOUNT_ID: SecretStr
    CENTRAL_R2_WRITE_ACCESS_KEY_ID: SecretStr
    CENTRAL_R2_WRITE_SECRET_ACCESS_KEY: SecretStr
    CENTRAL_R2_CONCURRENCY: int
    CENTRAL_R2_RESULTS_PREFIX: str

    # Audit Validator Cloudflare R2
    AUDIT_R2_BUCKET: str
    AUDIT_R2_ACCOUNT_ID: SecretStr
    AUDIT_R2_WRITE_ACCESS_KEY_ID: SecretStr
    AUDIT_R2_WRITE_SECRET_ACCESS_KEY: SecretStr
    AUDIT_R2_CONCURRENCY: int
    AUDIT_R2_BUCKET_PUBLIC_URL: str
    AUDIT_R2_RESULTS_PREFIX: str

    # Signer
    SIGNER_URL: str
    SIGNER_SEED: SecretStr
    SIGNER_HOST: str
    SIGNER_PORT: int

    # ScoreVision
    SCOREVISION_NETUID: int
    SCOREVISION_MECHID: int
    SCOREVISION_VERSION: str
    SCOREVISION_API: str
    SCOREVISION_VIDEO_FRAMES_PER_SECOND: int
    SCOREVISION_VIDEO_MIN_FRAME_NUMBER: int
    SCOREVISION_VIDEO_MAX_FRAME_NUMBER: int
    SCOREVISION_IMAGE_JPEG_QUALITY: int
    SCOREVISION_IMAGE_HEIGHT: int
    SCOREVISION_IMAGE_WIDTH: int
    SCOREVISION_VLM_SELECT_N_FRAMES: int
    SCOREVISION_VLM_TEMPERATURE: float
    SCOREVISION_API_TIMEOUT_S: int
    SCOREVISION_API_RETRY_DELAY_S: int
    SCOREVISION_API_N_RETRIES: int
    SCOREVISION_LOCAL_ROOT: Path
    SCOREVISION_WARMUP_CALLS: int
    SCOREVISION_MAX_CONCURRENT_API_CALLS: int
    SCOREVISION_BACKOFF_RATE: float
    SCOREVISION_TAIL: int
    SCOREVISION_M_MIN: int
    SCOREVISION_TEMPO: int
    SCOREVISION_CACHE_DIR: Path
    SCOREVISION_WINDOW_TIEBREAK_ENABLE: bool
    SCOREVISION_WINDOW_K_PER_VALIDATOR: int
    SCOREVISION_WINDOW_DELTA_ABS: float
    SCOREVISION_WINDOW_DELTA_REL: float
    SCOREVISION_WINDOW_HALF_LIFE: float

    # Runner
    RUNNER_GET_BLOCK_TIMEOUT_S: float
    RUNNER_WAIT_BLOCK_TIMEOUT_S: float
    RUNNER_RECONNECT_DELAY_S: float
    RUNNER_DEFAULT_ELEMENT_TEMPO: int
    RUNNER_PGT_MAX_BBOX_RETRIES: int
    RUNNER_PGT_MAX_QUALITY_RETRIES: int

    # Bittensor
    BLOCKS_PER_DAY: int

    # Validator
    VALIDATOR_TAIL_BLOCKS: int
    VALIDATOR_FALLBACK_UID: int
    VALIDATOR_WINNERS_EVERY_N: int

    # Audit Validator
    AUDIT_SPOTCHECK_MIN_INTERVAL_S: int
    AUDIT_SPOTCHECK_MAX_INTERVAL_S: int
    AUDIT_SPOTCHECK_THRESHOLD: float
    AUDIT_COMMIT_MAX_RETRIES: int
    AUDIT_COMMIT_RETRY_DELAY_S: float


def _env_bool(name: str, default: bool) -> bool:
    v = getenv(name, str(default))
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    central_results_prefix = getenv("CENTRAL_R2_RESULTS_PREFIX", "results_soccer")
    return Settings(
        # Bittensor
        BITTENSOR_WALLET_COLD=getenv("BITTENSOR_WALLET_COLD", "default"),
        BITTENSOR_WALLET_HOT=getenv("BITTENSOR_WALLET_HOT", "default"),
        BITTENSOR_WALLET_PATH=Path(
            getenv(
                "BITTENSOR_WALLET_PATH",
                str(Path.home() / ".bittensor" / "wallets"),
            )
        ).expanduser(),
        BITTENSOR_SUBTENSOR_ENDPOINT=getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney"),
        BITTENSOR_SUBTENSOR_FALLBACK=getenv(
            "BITTENSOR_SUBTENSOR_FALLBACK", "wss://entrypoint-finney.opentensor.ai:443"
        ),
        # Chutes
        CHUTES_USERNAME=getenv("CHUTES_USERNAME", ""),
        CHUTES_VLM=getenv("CHUTES_VLM", "Qwen/Qwen2.5-VL-72B-Instruct"),
        CHUTES_VLM_ENDPOINT=getenv(
            "CHUTES_VLM_ENDPOINT", "https://llm.chutes.ai/v1/chat/completions"
        ),
        CHUTES_SAM3_ENDPOINT=getenv(
            "CHUTES_SAM3_ENDPOINT", "https://score-test-sam3.chutes.ai/sam3/segment"
        ),
        CHUTES_SAM3_ID=getenv("CHUTES_SAM3_ID", "e97a02fe-1932-5f43-84b5-5dced5443012"),
        CHUTES_MINERS_ENDPOINT=getenv(
            "CHUTES_MINERS_ENDPOINT", "https://api.chutes.ai"
        ),
        CHUTES_MINER_PREDICT_ENDPOINT=getenv(
            "CHUTES_MINER_PREDICT_ENDPOINT", "predict"
        ),
        CHUTES_MINER_BASE_URL_TEMPLATE=getenv(
            "CHUTES_MINER_BASE_URL_TEMPLATE",
            "https://{slug}.chutes.ai",
        ),
        CHUTES_API_KEY=getenv("CHUTES_API_KEY", ""),
        PATH_CHUTE_TEMPLATES=Path(
            getenv("PATH_CHUTE_TEMPLATES", "scorevision/chute_template")
        ),
        PATH_CHUTE_SCRIPT=Path(
            getenv(
                "PATH_CHUTE_SCRIPT",
                "scorevision/chute_template/turbovision_chute.py.j2",
            )
        ),
        FILENAME_CHUTE_MAIN=getenv("FILENAME_CHUTE_MAIN", "chute.py.j2"),
        FILENAME_CHUTE_SCHEMAS=getenv("FILENAME_CHUTE_SCHEMAS", "schemas.py"),
        FILENAME_CHUTE_SETUP_UTILS=getenv("FILENAME_CHUTE_SETUP_UTILS", "setup.py"),
        FILENAME_CHUTE_LOAD_UTILS=getenv("FILENAME_CHUTE_LOAD_UTILS", "load.py"),
        FILENAME_CHUTE_PREDICT_UTILS=getenv(
            "FILENAME_CHUTE_PREDICT_UTILS", "predict.py"
        ),
        # OpenRouter
        OPENROUTER_API_KEY=getenv("OPENROUTER_API_KEY", ""),
        OPENROUTER_VLM_ENDPOINT=getenv(
            "OPENROUTER_VLM_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
        ),
        OPENROUTER_VLM=getenv("OPENROUTER_VLM", "qwen/qwen2.5-vl-72b-instruct:free"),
        # HuggingFace
        HUGGINGFACE_USERNAME=getenv("HUGGINGFACE_USERNAME", ""),
        HUGGINGFACE_API_KEY=getenv("HUGGINGFACE_API_KEY", ""),
        HUGGINGFACE_CONCURRENCY=int(getenv("HUGGINGFACE_CONCURRENCY", 2)),
        # Cloudflare R2
        SCOREVISION_BUCKET=getenv("SCOREVISION_BUCKET", ""),
        SCOREVISION_PUBLIC_RESULTS_URL=getenv("SCOREVISION_PUBLIC_RESULTS_URL", ""),
        CENTRAL_R2_ACCOUNT_ID=getenv("CENTRAL_R2_ACCOUNT_ID", ""),
        CENTRAL_R2_WRITE_ACCESS_KEY_ID=getenv("CENTRAL_R2_WRITE_ACCESS_KEY_ID", ""),
        CENTRAL_R2_WRITE_SECRET_ACCESS_KEY=getenv("CENTRAL_R2_WRITE_SECRET_ACCESS_KEY", ""),
        CENTRAL_R2_CONCURRENCY=int(getenv("CENTRAL_R2_CONCURRENCY", 8)),
        URL_MANIFEST=getenv("URL_MANIFEST", ""),
        AUDIT_R2_BUCKET=getenv("AUDIT_R2_BUCKET", ""),
        AUDIT_R2_ACCOUNT_ID=getenv("AUDIT_R2_ACCOUNT_ID", ""),
        AUDIT_R2_WRITE_ACCESS_KEY_ID=getenv("AUDIT_R2_WRITE_ACCESS_KEY_ID", ""),
        AUDIT_R2_WRITE_SECRET_ACCESS_KEY=getenv("AUDIT_R2_WRITE_SECRET_ACCESS_KEY", ""),
        AUDIT_R2_CONCURRENCY=int(getenv("AUDIT_R2_CONCURRENCY", 8)),
        AUDIT_R2_BUCKET_PUBLIC_URL=getenv("AUDIT_R2_BUCKET_PUBLIC_URL", ""),
        AUDIT_R2_RESULTS_PREFIX=getenv("AUDIT_R2_RESULTS_PREFIX", "audit_spotcheck"),
        CENTRAL_R2_RESULTS_PREFIX=central_results_prefix,
        # Signer
        SIGNER_URL=getenv("SIGNER_URL", "http://signer:8080"),
        SIGNER_SEED=getenv("SIGNER_SEED", ""),
        SIGNER_HOST=getenv("SIGNER_HOST", "127.0.0.1"),
        SIGNER_PORT=int(getenv("SIGNER_PORT", 8080)),
        # ScoreVision
        SCOREVISION_NETUID=int(getenv("SCOREVISION_NETUID", 44)),
        SCOREVISION_MECHID=1,
        SCOREVISION_VERSION=getenv("SCOREVISION_VERSION", __version__),
        SCOREVISION_API=getenv("SCOREVISION_API", "https://api.scorevision.io"),
        SCOREVISION_VIDEO_FRAMES_PER_SECOND=int(
            getenv("SCOREVISION_VIDEO_FRAMES_PER_SECOND", 30)
        ),
        SCOREVISION_VIDEO_MIN_FRAME_NUMBER=int(
            getenv("SCOREVISION_VIDEO_MIN_FRAME_NUMBER", 1)
        ),
        SCOREVISION_VIDEO_MAX_FRAME_NUMBER=int(
            getenv("SCOREVISION_VIDEO_MAX_FRAME_NUMBER", 750)
        ),
        SCOREVISION_IMAGE_JPEG_QUALITY=int(
            getenv("SCOREVISION_IMAGE_JPEG_QUALITY", 80)
        ),
        SCOREVISION_IMAGE_HEIGHT=int(getenv("SCOREVISION_IMAGE_HEIGHT", 540)),
        SCOREVISION_IMAGE_WIDTH=int(getenv("SCOREVISION_IMAGE_WIDTH", 960)),
        SCOREVISION_VLM_SELECT_N_FRAMES=int(
            getenv("SCOREVISION_VLM_SELECT_N_FRAMES", 3)
        ),
        SCOREVISION_VLM_TEMPERATURE=float(getenv("SCOREVISION_VLM_TEMPERATURE", 0.1)),
        SCOREVISION_API_TIMEOUT_S=int(getenv("SCOREVISION_API_TIMEOUT_S", 300)),
        SCOREVISION_API_RETRY_DELAY_S=int(getenv("SCOREVISION_API_RETRY_DELAY_S", 3)),
        SCOREVISION_API_N_RETRIES=int(getenv("SCOREVISION_API_N_RETRIES", 3)),
        SCOREVISION_LOCAL_ROOT=Path(
            getenv(
                "SCOREVISION_LOCAL_ROOT",
                Path.home() / ".cache" / "scorevision" / "local",
            )
        ),
        SCOREVISION_WARMUP_CALLS=int(getenv("SCOREVISION_WARMUP_CALLS", "3")),
        SCOREVISION_MAX_CONCURRENT_API_CALLS=int(
            getenv("SCOREVISION_MAX_CONCURRENT_API_CALLS", 8)
        ),
        SCOREVISION_BACKOFF_RATE=float(getenv("SCOREVISION_BACKOFF_RATE", 0.5)),
        SCOREVISION_TAIL=int(getenv("SCOREVISION_TAIL", 28800)),
        SCOREVISION_M_MIN=int(getenv("SCOREVISION_M_MIN", 25)),
        SCOREVISION_TEMPO=int(getenv("SCOREVISION_TEMPO", 100)),
        SCOREVISION_CACHE_DIR=Path(
            getenv("SCOREVISION_CACHE_DIR", "~/.cache/scorevision/blocks")
        ).expanduser(),
        SCOREVISION_WINDOW_TIEBREAK_ENABLE=_env_bool(
            "SCOREVISION_WINDOW_TIEBREAK_ENABLE", True
        ),
        SCOREVISION_WINDOW_K_PER_VALIDATOR=int(
            getenv("SCOREVISION_WINDOW_K_PER_VALIDATOR", 25)
        ),
        SCOREVISION_WINDOW_DELTA_ABS=float(
            getenv("SCOREVISION_WINDOW_DELTA_ABS", 0.003)
        ),
        SCOREVISION_WINDOW_DELTA_REL=float(
            getenv("SCOREVISION_WINDOW_DELTA_REL", 0.01)
        ),
        SCOREVISION_WINDOW_HALF_LIFE=float(getenv("SCOREVISION_WINDOW_HALF_LIFE", 3.0)),
        # Runner
        RUNNER_GET_BLOCK_TIMEOUT_S=float(getenv("SUBTENSOR_GET_BLOCK_TIMEOUT_S", 15.0)),
        RUNNER_WAIT_BLOCK_TIMEOUT_S=float(getenv("SUBTENSOR_WAIT_BLOCK_TIMEOUT_S", 15.0)),
        RUNNER_RECONNECT_DELAY_S=float(getenv("SUBTENSOR_RECONNECT_DELAY_S", 5.0)),
        RUNNER_DEFAULT_ELEMENT_TEMPO=int(getenv("SV_DEFAULT_ELEMENT_TEMPO_BLOCKS", 300)),
        RUNNER_PGT_MAX_BBOX_RETRIES=int(getenv("SV_PGT_MAX_BBOX_RETRIES", 3)),
        RUNNER_PGT_MAX_QUALITY_RETRIES=int(getenv("SV_PGT_MAX_QUALITY_RETRIES", 4)),
        # Bittensor
        BLOCKS_PER_DAY=7200,
        # Validator
        VALIDATOR_TAIL_BLOCKS=int(getenv("SCOREVISION_VALIDATOR_TAIL", 28800)),
        VALIDATOR_FALLBACK_UID=int(getenv("SCOREVISION_FALLBACK_UID", 6)),
        VALIDATOR_WINNERS_EVERY_N=int(getenv("SCOREVISION_WINNERS_EVERY", 24)),
        # Audit Validator
        AUDIT_SPOTCHECK_MIN_INTERVAL_S=int(getenv("AUDIT_SPOTCHECK_MIN_INTERVAL_S", 7200)),
        AUDIT_SPOTCHECK_MAX_INTERVAL_S=int(getenv("AUDIT_SPOTCHECK_MAX_INTERVAL_S", 14400)),
        AUDIT_SPOTCHECK_THRESHOLD=float(getenv("AUDIT_SPOTCHECK_THRESHOLD", 0.95)),
        AUDIT_COMMIT_MAX_RETRIES=int(getenv("AUDIT_COMMIT_MAX_RETRIES", 3)),
        AUDIT_COMMIT_RETRY_DELAY_S=float(getenv("AUDIT_COMMIT_RETRY_DELAY_S", 2.0)),
    )
