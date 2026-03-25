from pydantic import BaseModel, Field


class FramePrediction(BaseModel):
    frame: int = Field(ge=0)
    action: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChallengeRequest(BaseModel):
    challenge_id: str
    video_url: str


class ChallengeResponse(BaseModel):
    challenge_id: str
    predictions: list[FramePrediction] = Field(default_factory=list)
    processing_time: float
