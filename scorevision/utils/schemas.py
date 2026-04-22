from pydantic import BaseModel, Field, model_validator


class FramePrediction(BaseModel):
    frame: int = Field(ge=0)
    action: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChallengeFrame(BaseModel):
    frame_id: int = Field(ge=0)
    url: str | None = None
    data: str | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if not self.url and not self.data:
            raise ValueError("ChallengeFrame requires either url or data")
        return self


class ChallengeRequest(BaseModel):
    challenge_id: str
    video_url: str | None = None
    frames: list[ChallengeFrame] | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.video_url:
            return self
        if self.frames:
            return self
        raise ValueError("ChallengeRequest requires video_url or frames")


class ChallengeResponse(BaseModel):
    challenge_id: str
    predictions: list[FramePrediction] = Field(default_factory=list)
    processing_time: float
