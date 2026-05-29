from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.pillar_metric_registry import METRIC_REGISTRY
from scorevision.vlm_pipeline.non_vlm_scoring.polygons import (
    compare_polygon_counts,
    compare_polygon_false_positive,
    compare_polygon_placement,
    compare_polygon_precision,
    compare_polygon_recall,
)


def test_polygon_metrics_registered():
    assert (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_IOU) in METRIC_REGISTRY
    assert (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_COUNT) in METRIC_REGISTRY
    assert (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_PRECISION) in METRIC_REGISTRY
    assert (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_RECALL) in METRIC_REGISTRY
    assert (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_FALSE_POSITIVE) in METRIC_REGISTRY


def test_polygon_placeholder_metrics_return_zero():
    assert compare_polygon_placement() == 0.0
    assert compare_polygon_counts() == 0.0
    assert compare_polygon_precision() == 0.0
    assert compare_polygon_recall() == 0.0
    assert compare_polygon_false_positive() == 0.0
