from pytest import fixture

from numpy import zeros, ndarray

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    BoundingBox,
    ShirtColor,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Person, Action


@fixture
def fake_vlm_bbox() -> BoundingBox:
    return BoundingBox(
        bbox_2d=[10, 23, 100, 200], label=Person.BALL, cluster_id=ShirtColor.OTHER
    )


@fixture
def fake_annotation(fake_vlm_bbox) -> FrameAnnotation:
    return FrameAnnotation(
        bboxes=[fake_vlm_bbox, fake_vlm_bbox, fake_vlm_bbox],
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
