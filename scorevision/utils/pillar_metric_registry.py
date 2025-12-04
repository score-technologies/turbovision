from scorevision.utils.manifest import Manifest, ElementPrefix, PillarName

METRIC_REGISTRY: dict[tuple[ElementPrefix, PillarName], callable] = {}


def register_metric(*registrations: tuple[ElementPrefix, PillarName]):
    def wrapper(fn: callable):
        for element_prefix, pillar in registrations:
            METRIC_REGISTRY[(element_prefix, pillar)] = fn
        return fn

    return wrapper


def element_pillar_registry_availability() -> dict:
    return dict(
        elements=[
            dict(
                element_name=element.value,
                pillars=[
                    pillar.value
                    for pillar in PillarName
                    if (element, pillar) in METRIC_REGISTRY
                ],
            )
            for element in ElementPrefix
        ]
    )
