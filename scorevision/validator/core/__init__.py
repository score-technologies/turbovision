from importlib import import_module

__all__ = [
    "weights_loop",
    "set_weights_via_signer",
    "load_manifest_for_block",
    "commit_validator_on_start",
    "get_validator_hotkey_ss58",
    "run_signer",
]

_EXPORTS = {
    "weights_loop": ("scorevision.validator.core.weights", "weights_loop"),
    "set_weights_via_signer": ("scorevision.validator.core.weights", "set_weights_via_signer"),
    "load_manifest_for_block": ("scorevision.validator.core.weights", "load_manifest_for_block"),
    "commit_validator_on_start": ("scorevision.validator.core.weights", "commit_validator_on_start"),
    "get_validator_hotkey_ss58": ("scorevision.validator.core.weights", "get_validator_hotkey_ss58"),
    "run_signer": ("scorevision.validator.core.signer", "run_signer"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
