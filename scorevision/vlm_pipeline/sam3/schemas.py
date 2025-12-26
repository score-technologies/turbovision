from pydantic import BaseModel


class Polygons(BaseModel):
    masks: list[list[tuple[int, int]]]
    confidence: float


class ObjectName(BaseModel):
    text: str
    num_boxes: int


class Sam3Result(BaseModel):
    prompt_index: int
    echo: ObjectName
    predictions: list[Polygons]
