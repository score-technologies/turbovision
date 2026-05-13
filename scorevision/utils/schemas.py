from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator


class FramePrediction(BaseModel):
    frame: int = Field(ge=0)
    action: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CricketDeliveryPrediction(BaseModel):
    """Canonical cricket delivery payload."""

    model_config = ConfigDict(populate_by_name=True)

    match: str | None = None
    matchid: int | None = None
    inningsid: int | None = Field(default=None, validation_alias=AliasChoices("inningsid", "innings_id"))
    overid: int | None = Field(default=None, validation_alias=AliasChoices("overid", "over_id"))
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


class PredictionPayload(BaseModel):
    type: str
    items: list[FramePrediction] | None = None
    item: CricketDeliveryPrediction | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.type == "soccer_action":
            if self.items is None:
                raise ValueError("soccer_action prediction requires items")
            if self.item is not None:
                raise ValueError("soccer_action prediction must not include item")
            return self
        if self.type == "cricket_delivery":
            if self.item is None:
                raise ValueError("cricket_delivery prediction requires item")
            if self.items is not None:
                raise ValueError("cricket_delivery prediction must not include items")
            return self
        raise ValueError(f"Unsupported prediction type: {self.type}")


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
    prediction: PredictionPayload | CricketDeliveryPrediction | None = None
    processing_time: float

    @model_validator(mode="after")
    def validate_payload(self):
        if isinstance(self.prediction, CricketDeliveryPrediction):
            self.prediction = PredictionPayload(type="cricket_delivery", item=self.prediction)

        has_legacy_predictions = self.predictions is not None
        has_prediction = self.prediction is not None

        if not has_legacy_predictions and not has_prediction:
            raise ValueError("ChallengeResponse requires prediction payload")

        if self.prediction is None and self.predictions is not None:
            self.prediction = PredictionPayload(type="soccer_action", items=self.predictions)
        elif self.predictions is None and isinstance(self.prediction, PredictionPayload):
            if self.prediction.type == "soccer_action":
                self.predictions = self.prediction.items
        return self

    @property
    def prediction_count(self) -> int:
        if isinstance(self.prediction, PredictionPayload) and self.prediction.type == "cricket_delivery":
            return 1
        if isinstance(self.prediction, PredictionPayload) and self.prediction.type == "soccer_action":
            return len(self.prediction.items or [])
        return len(self.predictions or [])

    @property
    def is_cricket(self) -> bool:
        return isinstance(self.prediction, PredictionPayload) and self.prediction.type == "cricket_delivery"

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        data = handler(self)
        if isinstance(self.prediction, PredictionPayload):
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
