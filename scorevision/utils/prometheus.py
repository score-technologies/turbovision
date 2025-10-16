# ---- Prometheus ----
import os
from pathlib import Path
from prometheus_client import Counter, Gauge, CollectorRegistry, start_http_server
from scorevision.utils.settings import get_settings

settings = get_settings()
CACHE_DIR = settings.SCOREVISION_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROM_REG = CollectorRegistry(auto_describe=True)

SHARDS_READ_TOTAL = Counter(
    "shards_read_total", "Total shard lines read (raw)", registry=PROM_REG
)
SHARDS_VALID_TOTAL = Counter(
    "shards_valid_total", "Total shard lines passed validation", registry=PROM_REG
)
EMA_BY_UID = Gauge("ema_by_uid", "EMA score by uid", ["uid"], registry=PROM_REG)
WEIGHT_BY_UID = Gauge("weights", "Current weight by uid", ["uid"], registry=PROM_REG)
RANK_BY_UID = Gauge("rank", "Current rank by uid (1=best)", ["uid"], registry=PROM_REG)
CURRENT_WINNER = Gauge("current_winner_uid", "UID of current winner", registry=PROM_REG)
LASTSET_GAUGE = Gauge(
    "lastset", "Unix time of last successful set_weights", registry=PROM_REG
)
PREDICT_COUNT = Counter(
    "predict_count", "Predict calls counted from shards", ["model"], registry=PROM_REG
)
INDEX_KEYS_COUNT = Gauge(
    "index_keys_count", "Number of keys in index", registry=PROM_REG
)
CACHE_FILES = Gauge("cache_files", "Cached shard jsonl files", registry=PROM_REG)


SERVICE_INFO = Gauge(
    "scorevision_service_info",
    "Metrics service heartbeat",
    ["service"],
    registry=PROM_REG,
)

VALIDATOR_BLOCK_HEIGHT = Gauge(
    "validator_block_height",
    "Latest block observed by validator loop",
    registry=PROM_REG,
)
VALIDATOR_LOOP_TOTAL = Counter(
    "validator_loop_total",
    "Validator loop outcomes",
    ["outcome"],
    registry=PROM_REG,
)
VALIDATOR_LAST_BLOCK_SUCCESS = Gauge(
    "validator_last_block_success",
    "Last block height where weights set successfully",
    registry=PROM_REG,
)
VALIDATOR_WEIGHT_FAIL_TOTAL = Counter(
    "validator_weight_fail_total",
    "Validator weight submission failures",
    ["stage"],
    registry=PROM_REG,
)
VALIDATOR_CACHE_BYTES = Gauge(
    "validator_cache_bytes",
    "Total bytes of cached shard files",
    registry=PROM_REG,
)
VALIDATOR_DATASET_LINES_TOTAL = Counter(
    "validator_dataset_lines_total",
    "Dataset lines processed",
    ["source", "result"],
    registry=PROM_REG,
)
VALIDATOR_DATASET_FETCH_ERRORS_TOTAL = Counter(
    "validator_dataset_fetch_errors_total",
    "Dataset fetch errors",
    ["stage"],
    registry=PROM_REG,
)
VALIDATOR_MINERS_CONSIDERED = Gauge(
    "validator_miners_considered",
    "Number of miners passing robust filtering",
    registry=PROM_REG,
)
VALIDATOR_MINERS_SKIPPED_TOTAL = Counter(
    "validator_miners_skipped_total",
    "Validators skipped miners",
    ["reason"],
    registry=PROM_REG,
)
VALIDATOR_WINNER_SCORE = Gauge(
    "validator_winner_score",
    "Score for current winning miner",
    registry=PROM_REG,
)
VALIDATOR_LAST_LOOP_DURATION_SECONDS = Gauge(
    "validator_last_loop_duration_seconds",
    "Seconds taken by the most recent validator loop iteration",
    registry=PROM_REG,
)
VALIDATOR_SIGNER_REQUEST_DURATION_SECONDS = Gauge(
    "validator_signer_request_duration_seconds",
    "Seconds taken by the last validator signer request",
    registry=PROM_REG,
)
VALIDATOR_RECENT_WINDOW_SAMPLES = Gauge(
    "validator_recent_window_samples",
    "Recent window sample count per miner",
    ["uid"],
    registry=PROM_REG,
)
VALIDATOR_COMMIT_TOTAL = Counter(
    "validator_commit_total",
    "Validator commit attempts",
    ["result"],
    registry=PROM_REG,
)

RUNNER_BLOCK_HEIGHT = Gauge(
    "runner_block_height",
    "Latest block observed by runner loop",
    registry=PROM_REG,
)
RUNNER_RUNS_TOTAL = Counter(
    "runner_runs_total",
    "Runner loop outcomes",
    ["result"],
    registry=PROM_REG,
)
RUNNER_WARMUP_TOTAL = Counter(
    "runner_warmup_total",
    "Warmup attempts",
    ["result"],
    registry=PROM_REG,
)
RUNNER_PGT_RETRY_TOTAL = Counter(
    "runner_pgt_retry_total",
    "PGT generation retries",
    ["reason"],
    registry=PROM_REG,
)
RUNNER_PGT_FRAMES = Gauge(
    "runner_pgt_frames",
    "Frames retained after PGT quality filtering",
    registry=PROM_REG,
)
RUNNER_MINER_CALLS_TOTAL = Counter(
    "runner_miner_calls_total",
    "Miner call outcomes",
    ["outcome"],
    registry=PROM_REG,
)
RUNNER_MINER_LATENCY_MS = Gauge(
    "runner_miner_latency_ms",
    "Miner inference latency",
    ["miner"],
    registry=PROM_REG,
)
RUNNER_EVALUATION_SCORE = Gauge(
    "runner_evaluation_score",
    "Evaluation score by miner",
    ["miner"],
    registry=PROM_REG,
)
RUNNER_EVALUATION_FAIL_TOTAL = Counter(
    "runner_evaluation_fail_total",
    "Evaluation failures",
    ["stage"],
    registry=PROM_REG,
)
RUNNER_SHARDS_EMITTED_TOTAL = Counter(
    "runner_shards_emitted_total",
    "Shard emission results",
    ["status"],
    registry=PROM_REG,
)
RUNNER_ACTIVE_MINERS = Gauge(
    "runner_active_miners",
    "Active miners per runner invocation",
    registry=PROM_REG,
)
RUNNER_LAST_RUN_DURATION_SECONDS = Gauge(
    "runner_last_run_duration_seconds",
    "Seconds taken by the most recent runner invocation",
    registry=PROM_REG,
)
RUNNER_LAST_PGT_DURATION_SECONDS = Gauge(
    "runner_last_pgt_duration_seconds",
    "Seconds spent building PGT in the most recent runner invocation",
    registry=PROM_REG,
)
RUNNER_MINER_LAST_DURATION_SECONDS = Gauge(
    "runner_miner_last_duration_seconds",
    "Seconds spent scoring per miner in the most recent run",
    ["miner"],
    registry=PROM_REG,
)


def mark_service_ready(service: str) -> None:
    try:
        SERVICE_INFO.labels(service=service).set(1)
    except Exception:
        pass


def _start_metrics():
    if os.getenv("USE_PROMETHEUS", "true").lower() in ("0", "false", "no"):
        return
    try:
        port = int(os.getenv("SCOREVISION_METRICS_PORT", "8010"))
        addr = os.getenv("SCOREVISION_METRICS_ADDR", "0.0.0.0")
        start_http_server(port, addr, registry=PROM_REG)
    except Exception:
        pass
