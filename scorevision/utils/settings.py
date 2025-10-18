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

    # Cloudflare R2
    R2_BUCKET: str
    R2_ACCOUNT_ID: SecretStr
    R2_WRITE_ACCESS_KEY_ID: SecretStr
    R2_WRITE_SECRET_ACCESS_KEY: SecretStr
    R2_CONCURRENCY: int
    R2_BUCKET_PUBLIC_URL: str

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

    SCOREVISION_SCORE_BASE: float
    SCOREVISION_SCORE_ALPHA: float
    SCOREVISION_SCORE_MS_PENALTY: float
    SCOREVISION_SCORE_SERVICE_LEVEL_OBJECTIVE_MS: float

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

    SCOREVISION_RESULTS_PREFIX: str
    SCOREVISION_LOCAL_ROOT: Path
    SCOREVISION_WARMUP_CALLS: int
    SCOREVISION_MAX_CONCURRENT_API_CALLS: int
    SCOREVISION_BACKOFF_RATE: float
    SCOREVISION_TAIL: int
    SCOREVISION_ALPHA: float
    SCOREVISION_M_MIN: int
    SCOREVISION_TEMPO: int
    SCOREVISION_CACHE_DIR: Path
    SCOREVISION_WINDOW_TIEBREAK_ENABLE: bool
    SCOREVISION_WINDOW_K_PER_VALIDATOR: int
    SCOREVISION_WINDOW_DELTA_ABS: float
    SCOREVISION_WINDOW_DELTA_REL: float


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # No defaults - MUST be set by User
        SCOREVISION_API=getenv("SCOREVISION_API", "https://api.scorevision.io"),
        CHUTES_API_KEY=getenv("CHUTES_API_KEY", ""),
        BITTENSOR_WALLET_PATH=Path(
            getenv(
                "BITTENSOR_WALLET_PATH",
                str(Path.home() / ".bittensor" / "wallets"),
            )
        ).expanduser(),
        OPENROUTER_API_KEY=getenv("OPENROUTER_API_KEY", ""),
        R2_BUCKET=getenv("R2_BUCKET", ""),
        R2_ACCOUNT_ID=getenv("R2_ACCOUNT_ID", ""),
        R2_WRITE_ACCESS_KEY_ID=getenv("R2_WRITE_ACCESS_KEY_ID", ""),
        R2_WRITE_SECRET_ACCESS_KEY=getenv("R2_WRITE_SECRET_ACCESS_KEY", ""),
        R2_BUCKET_PUBLIC_URL=getenv("R2_BUCKET_PUBLIC_URL", ""),
        HUGGINGFACE_USERNAME=getenv("HUGGINGFACE_USERNAME", ""),
        HUGGINGFACE_API_KEY=getenv("HUGGINGFACE_API_KEY", ""),
        CHUTES_USERNAME=getenv("CHUTES_USERNAME", ""),
        SIGNER_SEED=getenv("SIGNER_SEED", ""),
        # Defaults - CAN be changed by User
        BITTENSOR_WALLET_COLD=getenv("BITTENSOR_WALLET_COLD", "default"),
        BITTENSOR_WALLET_HOT=getenv("BITTENSOR_WALLET_HOT", "default"),
        SCOREVISION_NETUID=int(getenv("SCOREVISION_NETUID", 44)),
        SCOREVISION_MECHID=1,
        SCOREVISION_MAX_CONCURRENT_API_CALLS=int(
            getenv("SCOREVISION_MAX_CONCURRENT_API_CALLS", 8)
        ),
        SCOREVISION_BACKOFF_RATE=float(getenv("SCOREVISION_BACKOFF_RATE", 0.5)),
        SCOREVISION_VIDEO_FRAMES_PER_SECOND=int(
            getenv("SCOREVISION_VIDEO_FRAMES_PER_SECOND", 30)
        ),
        SCOREVISION_IMAGE_JPEG_QUALITY=int(
            getenv("SCOREVISION_IMAGE_JPEG_QUALITY", 80)
        ),
        SCOREVISION_VLM_SELECT_N_FRAMES=int(
            getenv("SCOREVISION_VLM_SELECT_N_FRAMES", 3)
        ),
        CHUTES_VLM_ENDPOINT=getenv(
            "CHUTES_VLM_ENDPOINT", "https://llm.chutes.ai/v1/chat/completions"
        ),
        OPENROUTER_VLM_ENDPOINT=getenv(
            "OPENROUTER_VLM_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
        ),
        CHUTES_VLM=getenv("CHUTES_VLM", "Qwen/Qwen2.5-VL-72B-Instruct"),
        OPENROUTER_VLM=getenv("OPENROUTER_VLM", "qwen/qwen2.5-vl-72b-instruct:free"),
        SCOREVISION_VLM_TEMPERATURE=float(getenv("SCOREVISION_VLM_TEMPERATURE", 0.1)),
        SCOREVISION_API_TIMEOUT_S=int(getenv("SCOREVISION_API_TIMEOUT_S", 300)),
        SCOREVISION_VIDEO_MIN_FRAME_NUMBER=int(
            getenv("SCOREVISION_VIDEO_MIN_FRAME_NUMBER", 1)
        ),
        SCOREVISION_VIDEO_MAX_FRAME_NUMBER=int(
            getenv("SCOREVISION_VIDEO_MAX_FRAME_NUMBER", 750)
        ),
        SCOREVISION_API_RETRY_DELAY_S=int(getenv("SCOREVISION_API_RETRY_DELAY_S", 3)),
        SCOREVISION_API_N_RETRIES=int(getenv("SCOREVISION_API_N_RETRIES", 3)),
        SCOREVISION_SCORE_BASE=float(
            getenv("SCOREVISION_SCORE_BASE", 2.718281828459045)
        ),
        SCOREVISION_SCORE_ALPHA=float(getenv("SCOREVISION_SCORE_ALPHA", 0.9)),
        SCOREVISION_SCORE_MS_PENALTY=float(
            getenv("SCOREVISION_SCORE_MS_PENALTY", 0.001)
        ),
        SCOREVISION_SCORE_SERVICE_LEVEL_OBJECTIVE_MS=float(
            getenv("SCOREVISION_SCORE_SERVICE_LEVEL_OBJECTIVE_MS", 1500.0)
        ),
        SCOREVISION_VERSION=getenv("SCOREVISION_VERSION", __version__),
        SCOREVISION_RESULTS_PREFIX=getenv(
            "SCOREVISION_RESULTS_PREFIX", "results_soccer"
        ),
        SCOREVISION_LOCAL_ROOT=Path(
            getenv(
                "SCOREVISION_LOCAL_ROOT",
                Path.home() / ".cache" / "scorevision" / "local",
            )
        ),
        R2_CONCURRENCY=int(getenv("R2_CONCURRENCY", 8)),
        HUGGINGFACE_CONCURRENCY=int(getenv("HUGGINGFACE_CONCURRENCY", 2)),
        PATH_CHUTE_SCRIPT=Path(
            getenv(
                "PATH_CHUTE_SCRIPT",
                "scorevision/chute_template/turbovision_chute.py.j2",
            )
        ),
        PATH_CHUTE_TEMPLATES=Path(
            getenv("PATH_CHUTE_TEMPLATES", "scorevision/chute_template")
        ),
        FILENAME_CHUTE_MAIN=getenv("FILENAME_CHUTE_MAIN", "chute.py.j2"),
        FILENAME_CHUTE_SCHEMAS=getenv("FILENAME_CHUTE_SCHEMAS", "schemas.py"),
        FILENAME_CHUTE_SETUP_UTILS=getenv("FILENAME_CHUTE_SETUP_UTILS", "setup.py"),
        FILENAME_CHUTE_LOAD_UTILS=getenv("FILENAME_CHUTE_LOAD_UTILS", "load.py"),
        FILENAME_CHUTE_PREDICT_UTILS=getenv(
            "FILENAME_CHUTE_PREDICT_UTILS", "predict.py"
        ),
        CHUTES_MINERS_ENDPOINT=getenv(
            "CHUTES_MINERS_ENDPOINT", "https://api.chutes.ai"
        ),
        CHUTES_MINER_BASE_URL_TEMPLATE=getenv(
            "CHUTES_MINER_BASE_URL_TEMPLATE",
            "https://{slug}.chutes.ai",
        ),
        CHUTES_MINER_PREDICT_ENDPOINT=getenv(
            "CHUTES_MINER_PREDICT_ENDPOINT", "predict"
        ),
        BITTENSOR_SUBTENSOR_ENDPOINT=getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney"),
        BITTENSOR_SUBTENSOR_FALLBACK=getenv(
            "BITTENSOR_SUBTENSOR_FALLBACK", "wss://entrypoint-finney.opentensor.ai:443"
        ),
        SCOREVISION_WARMUP_CALLS=int(getenv("SCOREVISION_WARMUP_CALLS", "3")),
        SIGNER_HOST=getenv("SIGNER_HOST", "127.0.0.1"),
        SIGNER_PORT=int(getenv("SIGNER_PORT", 8080)),
        SIGNER_URL=getenv("SIGNER_URL", "http://signer:8080"),
        SCOREVISION_TAIL=int(getenv("SCOREVISION_TAIL", 28800)),
        SCOREVISION_ALPHA=float(getenv("SCOREVISION_ALPHA", 0.2)),
        SCOREVISION_M_MIN=int(getenv("SCOREVISION_M_MIN", 25)),
        SCOREVISION_TEMPO=int(getenv("SCOREVISION_TEMPO", 100)),
        SCOREVISION_IMAGE_HEIGHT=int(getenv("SCOREVISION_IMAGE_HEIGHT", 540)),
        SCOREVISION_IMAGE_WIDTH=int(getenv("SCOREVISION_IMAGE_WIDTH", 960)),
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
    )


def _env_bool(name: str, default: bool) -> bool:
    v = getenv(name, str(default))
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")
