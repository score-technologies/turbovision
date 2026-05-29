from scorevision.utils.manifest import (
    KEYPOINT_TEMPLATES,
    ElementPrefix,
    Manifest,
    PillarName,
)

METRIC_REGISTRY: dict[tuple[ElementPrefix, PillarName], callable] = {}


def register_metric(*registrations: tuple[ElementPrefix, PillarName]):
    def wrapper(fn: callable):
        for element_prefix, pillar in registrations:
            METRIC_REGISTRY[(element_prefix, pillar)] = fn
        return fn

    return wrapper


def element_pillar_registry_availability() -> dict:
    element_metadatas = []
    for element in ElementPrefix:
        pillars = [
            pillar.value
            for pillar in PillarName
            if (element, pillar) in METRIC_REGISTRY
        ]
        if not any(pillars):
            continue
        element_metadata = dict(element_name=element.value, pillars=pillars)
        if element.value == ElementPrefix.PITCH_CALIBRATION:
            element_metadata["keypoint_template"] = list(KEYPOINT_TEMPLATES)
        elif element.value in (
            ElementPrefix.OBJECT_DETECTION,
            ElementPrefix.PLAYER_DETECTION,
        ):
            element_metadata["objects"] = [
                "object name 1 (e.g. player)",
                "object name 2 (e.g. ball)",
                "...",
            ]
        element_metadatas.append(element_metadata)
    return dict(element=element_metadatas)
