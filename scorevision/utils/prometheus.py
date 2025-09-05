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


def _start_metrics():
    try:
        port = int(os.getenv("SCOREVISION_METRICS_PORT", "8010"))
        addr = os.getenv("SCOREVISION_METRICS_ADDR", "0.0.0.0")
        start_http_server(port, addr, registry=PROM_REG)
    except Exception:
        pass
