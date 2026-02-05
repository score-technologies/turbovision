from dataclasses import dataclass
from typing import Any


@dataclass
class ChallengeRecord:
    challenge_id: str
    element_id: str
    window_id: str
    block: int
    miner_hotkey: str
    central_score: float
    payload: dict[str, Any]
    miner_predictions: dict[str, Any] | None = None
    video_url: str | None = None
    responses_key: str | None = None
    scored_frame_numbers: list[int] | None = None


@dataclass
class SpotcheckResult:
    challenge_id: str
    element_id: str
    miner_hotkey: str
    central_score: float
    audit_score: float
    match_percentage: float
    passed: bool
    details: dict[str, Any] | None = None


@dataclass
class WeightsResult:
    element_id: str
    window_id: str
    winner_uid: int | None
    scores_by_uid: dict[int, float]
    winner_meta: dict[str, str | None] | None


@dataclass
class MinerMeta:
    hotkey: str
    chute_id: str | None = None
    slug: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "hotkey": self.hotkey,
            "chute_id": self.chute_id,
            "slug": self.slug,
        }

