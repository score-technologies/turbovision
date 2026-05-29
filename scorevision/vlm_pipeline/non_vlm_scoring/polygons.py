from logging import getLogger

from scorevision.utils.manifest import ElementPrefix, PillarName
from scorevision.utils.pillar_metric_registry import register_metric

logger = getLogger(__name__)


def _polygon_placeholder_score(name: str) -> float:
    logger.warning("Polygon scoring placeholder invoked for %s", name)
    return 0.0


@register_metric(
    (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_IOU),
)
def compare_polygon_placement(*args, **kwargs) -> float:
    return _polygon_placeholder_score("polygon_iou")


@register_metric(
    (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_COUNT),
)
def compare_polygon_counts(*args, **kwargs) -> float:
    return _polygon_placeholder_score("polygon_count")


@register_metric(
    (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_PRECISION),
)
def compare_polygon_precision(*args, **kwargs) -> float:
    return _polygon_placeholder_score("polygon_precision")


@register_metric(
    (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_RECALL),
)
def compare_polygon_recall(*args, **kwargs) -> float:
    return _polygon_placeholder_score("polygon_recall")


@register_metric(
    (ElementPrefix.POLYGON_DETECTION, PillarName.POLYGON_FALSE_POSITIVE),
)
def compare_polygon_false_positive(*args, **kwargs) -> float:
    return _polygon_placeholder_score("polygon_false_positive")
