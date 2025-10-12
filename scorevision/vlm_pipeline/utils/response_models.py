from enum import Enum

from pydantic import BaseModel, Field

from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action, ShirtColor


TEAM1_SHIRT_COLOUR = (
    ShirtColor.WHITE
)  # arbitrary colours to convert team 1/2 predictions into distinct colours
TEAM2_SHIRT_COLOUR = ShirtColor.BLACK


class BoundingBox(BaseModel):
    """The bounding box around a single object"""

    bbox_2d: tuple[int, int, int, int] = Field(
        ...,
        description="x_min, y_min, x_max, y_max in pixels (where x=0 is the left of the image and y=0 is the top of the image",
    )
    label: ObjectOfInterest = Field(..., description="The type of object shown")
    cluster_id: ShirtColor = Field(
        ...,
        description="Based on the visual appearance and colours of the object, assign a cluster colour to group it with other similar looking objects.",
    )


class FrameAnnotation(BaseModel):
    """Annotations for objects and points of interest shown in a video frame"""

    bboxes: list[BoundingBox] = Field(
        ..., description="The objects of interest in this frame"
    )
    category: Action = Field(..., description="The action being shown in this frame")
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score between 0 and 100 for the category predicted",
    )
    reason: str = Field(..., description="Reasoning for the category predicted")


class Winner(Enum):
    PSEUDO_GT = "Image A"
    MINER = "Image B"
    TIE = "Tie"


class VLMJudgeResult(BaseModel):
    thought: str = Field(
        ...,
        description="The step-by-step reasoning process used to analyse the question and images",
    )
    justification: str = Field(
        ...,
        description="Explanation for the judgment, detailing key factors that led to the conclusion",
    )
    winner: Winner = Field(..., description="Which is better?")


class VLMJudgeFrameResults(BaseModel):
    detections: VLMJudgeResult = Field(
        ...,
        description="Were all the objects of interest detected or were some missed?",
    )
    bboxes: VLMJudgeResult = Field(
        ...,
        description="Were the bounding boxes exactly around the objects of interest?",
    )
    labels: VLMJudgeResult = Field(
        ..., description="Were all the objects correctly labelled?"
    )
    category: VLMJudgeResult = Field(
        ...,
        description="Which categorisation more accurately describes what is happening in the image?",
    )
