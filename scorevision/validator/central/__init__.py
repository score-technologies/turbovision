from importlib import import_module

__all__ = [
    "runner",
    "runner_loop",
]

_EXPORTS = {
    "runner": ("scorevision.validator.central.open_source.runner", "runner"),
    "runner_loop": ("scorevision.validator.central.open_source.runner", "runner_loop"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
