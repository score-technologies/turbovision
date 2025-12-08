from scorevision.utils.manifest import Manifest, ElementPrefix, PillarName

METRIC_REGISTRY: dict[tuple[ElementPrefix, PillarName], callable] = {}


def register_metric(element_prefix: ElementPrefix, pillar: PillarName):
    def wrapper(fn: callable):
        METRIC_REGISTRY[(element_prefix, pillar)] = fn
        return fn

    return wrapper


def element_pillar_registry_availability() -> dict:
    return dict(
        elements=[
            dict(
                element_name=element.value,
                pillars=[
                    dict(
                        pillar_name=pillar.value,
                        metric_assigned=(element, pillar) in METRIC_REGISTRY,
                    )
                    for pillar in PillarName
                ],
            )
            for element in ElementPrefix
        ]
    )
