import json
import os
from dataclasses import dataclass
from scorevision.utils.schemas import FramePrediction


@dataclass(frozen=True)
class RunnerConfig:
    challenge_id: str
    video_url: str
    ground_truth: list[FramePrediction]
    original_score: float
    miner_hotkey: str
    match_threshold: float
    miner_timeout_s: float
    allowed_video_domains: frozenset[str]
    miner_url: str


def load_config() -> RunnerConfig:
    raw_domains = os.environ.get("ALLOWED_VIDEO_DOMAINS", "scoredata.me")
    allowed = frozenset(d.strip().lower() for d in raw_domains.split(",") if d.strip())

    return RunnerConfig(
        challenge_id=os.environ["CHALLENGE_ID"],
        video_url=os.environ["VIDEO_URL"],
        ground_truth=[
            FramePrediction(**gt)
            for gt in json.loads(os.environ["GROUND_TRUTH_JSON"])
        ],
        original_score=float(os.environ["ORIGINAL_SCORE"]),
        miner_hotkey=os.environ["MINER_HOTKEY"],
        match_threshold=float(os.environ.get("MATCH_THRESHOLD", "0.98")),
        miner_timeout_s=float(os.environ.get("MINER_TIMEOUT_S", "120")),
        allowed_video_domains=allowed,
        miner_url=os.environ.get("MINER_URL", "http://localhost:8000"),
    )
