_LAZY_EXPORTS = {
    "runner": ("scorevision.validator.central.open_source.runner", "runner"),
    "runner_loop": ("scorevision.validator.central.open_source.runner", "runner_loop"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "runner",
    "runner_loop",
]
