from pytest import fixture

from numpy import zeros, ndarray

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.geometry import (
    AnnotationGeometry,
    AnnotationGeometryType,
    Point2D,
)
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    BoundingBox,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action


@fixture
def fake_vlm_bbox() -> BoundingBox:
    return BoundingBox(
        label="ball",
        geometry=AnnotationGeometry(
            type=AnnotationGeometryType.BBOX,
            points=[Point2D(x=10, y=23), Point2D(x=100, y=200)],
        ),
    )


@fixture
def fake_annotation(fake_vlm_bbox) -> FrameAnnotation:
    return FrameAnnotation(
        annotations=[fake_vlm_bbox, fake_vlm_bbox, fake_vlm_bbox],
        category=Action.GOAL,
        confidence=100,
        reason="",
    )


@fixture
def dummy_pseudo_gt_annotations(fake_annotation) -> list[PseudoGroundTruth]:
    return [
        PseudoGroundTruth(
            video_name="test",
            frame_number=1,
            spatial_image=zeros((3, 3)),
            temporal_image=zeros((3, 3)),
            annotation=fake_annotation,
        ),
        PseudoGroundTruth(
            video_name="test",
            frame_number=2,
            spatial_image=zeros((3, 3)),
            temporal_image=zeros((3, 3)),
            annotation=fake_annotation,
        ),
        PseudoGroundTruth(
            video_name="test",
            frame_number=3,
            spatial_image=zeros((3, 3)),
            temporal_image=zeros((3, 3)),
            annotation=fake_annotation,
        ),
    ]
