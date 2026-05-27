_LAZY_EXPORTS = {
    "spotcheck_loop": ("scorevision.validator.audit.open_source.spotcheck", "spotcheck_loop"),
    "run_single_spotcheck": ("scorevision.validator.audit.open_source.spotcheck", "run_single_spotcheck"),
    "run_spotcheck": ("scorevision.validator.audit.open_source.spotcheck", "run_spotcheck"),
    "fetch_random_challenge_record": ("scorevision.validator.audit.open_source.spotcheck", "fetch_random_challenge_record"),
    "load_challenge_record_from_mock_dir": ("scorevision.validator.audit.open_source.spotcheck", "load_challenge_record_from_mock_dir"),
    "calculate_match_percentage": ("scorevision.validator.audit.open_source.spotcheck", "calculate_match_percentage"),
    "scores_match": ("scorevision.validator.audit.open_source.spotcheck", "scores_match"),
    "calculate_next_spotcheck_delay": ("scorevision.validator.audit.open_source.spotcheck", "calculate_next_spotcheck_delay"),
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
    "spotcheck_loop",
    "run_single_spotcheck",
    "run_spotcheck",
    "fetch_random_challenge_record",
    "load_challenge_record_from_mock_dir",
    "calculate_match_percentage",
    "scores_match",
    "calculate_next_spotcheck_delay",
]
