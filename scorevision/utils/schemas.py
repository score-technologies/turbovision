from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator


class FramePrediction(BaseModel):
    frame: int = Field(ge=0)
    action: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CricketDeliveryPrediction(BaseModel):
    """Canonical cricket delivery payload.

    The validator accepts the full GT-aligned row, but miners should focus first on:
    kph, bounce_x, stump_y, deviation, swing_angle, and stump_z.
    """

    model_config = ConfigDict(populate_by_name=True)

    match: str | None = None
    matchid: int | None = None
    inningsid: int | None = Field(default=None, validation_alias=AliasChoices("inningsid", "innings"))
    overid: int | None = Field(default=None, validation_alias=AliasChoices("overid", "over"))
    ball_in_over: int | None = Field(
        default=None,
        validation_alias=AliasChoices("ball_in_over", "ball"),
    )
    ballid: int | None = None
    xlsx_overs: str | float | None = None
    scorecard_overs: str | float | None = Field(
        default=None,
        validation_alias=AliasChoices("scorecard_overs", "overs"),
    )
    kph: float | None = None
    release_y: float | None = Field(default=None, validation_alias=AliasChoices("release_y", "rel_y"))
    release_z: float | None = Field(default=None, validation_alias=AliasChoices("release_z", "rel_z"))
    bounce_x: float | None = None
    bounce_y: float | None = None
    impact_x: float | None = None
    impact_y: float | None = None
    impact_z: float | None = None
    interception_distance: float | None = Field(
        default=None,
        validation_alias=AliasChoices("interception_distance", "inter_d"),
    )
    stump_y: float | None = None
    stump_z: float | None = None
    swing_angle: float | None = Field(
        default=None,
        validation_alias=AliasChoices("swing_angle", "swing_deg"),
    )
    deviation: float | None = Field(
        default=None,
        validation_alias=AliasChoices("deviation", "deviation_deg"),
    )
    runs: int | None = None
    wickets: int | None = Field(default=None, validation_alias=AliasChoices("wickets", "wkts"))


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
    predictions: list[FramePrediction] | None = None
    prediction: CricketDeliveryPrediction | None = None
    processing_time: float

    @model_validator(mode="after")
    def validate_payload(self):
        has_legacy_predictions = self.predictions is not None
        has_cricket_prediction = self.prediction is not None
        if has_legacy_predictions == has_cricket_prediction:
            raise ValueError("ChallengeResponse requires exactly one of predictions or prediction")
        return self

    @property
    def prediction_count(self) -> int:
        if self.prediction is not None:
            return 1
        return len(self.predictions or [])

    @property
    def is_cricket(self) -> bool:
        return self.prediction is not None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        data = handler(self)
        if self.is_cricket:
            return {
                "challenge_id": data["challenge_id"],
                "prediction": data["prediction"],
                "processing_time": data["processing_time"],
            }
        return {
            "challenge_id": data["challenge_id"],
            "predictions": data["predictions"] or [],
            "processing_time": data["processing_time"],
        }
