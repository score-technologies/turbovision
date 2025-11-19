from scorevision.utils.manifest import Manifest, ElementPrefix, PillarName

METRIC_REGISTRY: dict[tuple[ElementPrefix, PillarName], callable] = {}


def register_metric(element_prefix: ElementPrefix, pillar: PillarName):
    def wrapper(fn: callable):
        METRIC_REGISTRY[(element_prefix, pillar)] = fn
        return fn

    return wrapper
