from datetime import datetime
from pydantic import BaseModel


class PendingSpotcheck(BaseModel):
    datetime_spotcheck: datetime
    miner_hotkey: str
    miner_coldkey: str
    miner_username: str
    miner_image_repo: str
    miner_image_tag: str
    miner_image_digest: str
    scoring_version: int
    challenge_id: str
    challenge_url: str
    original_score: float
