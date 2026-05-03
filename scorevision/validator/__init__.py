from importlib import import_module

__all__ = [
    "ChallengeRecord",
    "OpenSourceMinerMeta",
    "SpotcheckResult",
    "WeightsResult",
    "build_winner_meta",
    "extract_challenge_id",
    "extract_elements_from_manifest",
    "extract_miner_and_score",
    "extract_miner_meta",
    "aggregate_challenge_scores_by_miner",
    "are_similar_by_challenges",
    "days_to_blocks",
    "pick_winner_with_tiebreak",
    "stake_of",
    "weighted_median",
    "get_winner_for_element",
    "weights_loop",
    "set_weights_via_signer",
    "load_manifest_for_block",
    "commit_validator_on_start",
    "get_validator_hotkey_ss58",
    "run_signer",
    "runner",
    "runner_loop",
    "spotcheck_loop",
    "run_single_spotcheck",
    "run_spotcheck",
    "fetch_random_challenge_record",
    "load_challenge_record_from_mock_dir",
    "calculate_match_percentage",
    "scores_match",
    "calculate_next_spotcheck_delay",
]

_EXPORTS = {
    "ChallengeRecord": ("scorevision.validator.models", "ChallengeRecord"),
    "OpenSourceMinerMeta": ("scorevision.validator.models", "OpenSourceMinerMeta"),
    "SpotcheckResult": ("scorevision.validator.models", "SpotcheckResult"),
    "WeightsResult": ("scorevision.validator.models", "WeightsResult"),
    "build_winner_meta": ("scorevision.validator.payload", "build_winner_meta"),
    "extract_challenge_id": ("scorevision.validator.payload", "extract_challenge_id"),
    "extract_elements_from_manifest": ("scorevision.validator.payload", "extract_elements_from_manifest"),
    "extract_miner_and_score": ("scorevision.validator.payload", "extract_miner_and_score"),
    "extract_miner_meta": ("scorevision.validator.payload", "extract_miner_meta"),
    "aggregate_challenge_scores_by_miner": ("scorevision.validator.scoring", "aggregate_challenge_scores_by_miner"),
    "are_similar_by_challenges": ("scorevision.validator.scoring", "are_similar_by_challenges"),
    "days_to_blocks": ("scorevision.validator.scoring", "days_to_blocks"),
    "pick_winner_with_tiebreak": ("scorevision.validator.scoring", "pick_winner_with_tiebreak"),
    "stake_of": ("scorevision.validator.scoring", "stake_of"),
    "weighted_median": ("scorevision.validator.scoring", "weighted_median"),
    "get_winner_for_element": ("scorevision.validator.winner", "get_winner_for_element"),
    "weights_loop": ("scorevision.validator.core", "weights_loop"),
    "set_weights_via_signer": ("scorevision.validator.core", "set_weights_via_signer"),
    "load_manifest_for_block": ("scorevision.validator.core", "load_manifest_for_block"),
    "commit_validator_on_start": ("scorevision.validator.core", "commit_validator_on_start"),
    "get_validator_hotkey_ss58": ("scorevision.validator.core", "get_validator_hotkey_ss58"),
    "run_signer": ("scorevision.validator.core", "run_signer"),
    "runner": ("scorevision.validator.central", "runner"),
    "runner_loop": ("scorevision.validator.central", "runner_loop"),
    "spotcheck_loop": ("scorevision.validator.audit", "spotcheck_loop"),
    "run_single_spotcheck": ("scorevision.validator.audit", "run_single_spotcheck"),
    "run_spotcheck": ("scorevision.validator.audit", "run_spotcheck"),
    "fetch_random_challenge_record": ("scorevision.validator.audit", "fetch_random_challenge_record"),
    "load_challenge_record_from_mock_dir": ("scorevision.validator.audit", "load_challenge_record_from_mock_dir"),
    "calculate_match_percentage": ("scorevision.validator.audit", "calculate_match_percentage"),
    "scores_match": ("scorevision.validator.audit", "scores_match"),
    "calculate_next_spotcheck_delay": ("scorevision.validator.audit", "calculate_next_spotcheck_delay"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
