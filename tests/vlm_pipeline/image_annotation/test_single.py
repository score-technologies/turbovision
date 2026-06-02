import numpy as np

from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.image_annotation.single import annotate_image
from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, ShirtColor


def test_annotate_image_renders_polygon_overlay():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    annotations = FrameAnnotation(
        bboxes=[
            BoundingBox(
                bbox_2d=(4, 4, 20, 20),
                polygon=[(4, 4), (20, 4), (20, 20), (4, 20)],
                label="ball",
                cluster_id=ShirtColor.RED,
            )
        ],
        category=Action.NONE,
        confidence=100,
        reason="test",
    )

    annotated = annotate_image(image=image, annotations=annotations, name="pgt")

    assert np.any(annotated != image)
