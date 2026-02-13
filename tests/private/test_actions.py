from scorevision.utils.actions import Action, ActionConfig, ACTION_CONFIGS


def test_all_actions_have_configs():
    for action in Action:
        assert action in ACTION_CONFIGS


def test_action_config_is_named_tuple():
    config = ACTION_CONFIGS[Action.PASS]
    assert isinstance(config, ActionConfig)
    assert config.weight == 1.0
    assert config.min_score == 0.0
    assert config.tolerance_seconds == 1.0


def test_action_string_values():
    assert Action.PASS.value == "pass"
    assert Action.GOAL.value == "goal"
    assert Action.END_OF_HALF.value == "end_of_half"


def test_action_from_string():
    assert Action("pass") == Action.PASS
    assert Action("goal") == Action.GOAL


def test_weights_are_positive():
    for action, config in ACTION_CONFIGS.items():
        assert config.weight > 0, f"{action} has non-positive weight"


def test_tolerances_are_positive():
    for action, config in ACTION_CONFIGS.items():
        assert config.tolerance_seconds > 0, f"{action} has non-positive tolerance"


def test_min_scores_in_valid_range():
    for action, config in ACTION_CONFIGS.items():
        assert 0.0 <= config.min_score <= 1.0, f"{action} has invalid min_score"
