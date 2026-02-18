from enum import Enum
from typing import NamedTuple


class Action(str, Enum):
    PASS = "pass"
    PASS_RECEIVED = "pass_received"
    RECOVERY = "recovery"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BALL_OUT_OF_PLAY = "ball_out_of_play"
    CLEARANCE = "clearance"
    DEFENSIVE_1ON1 = "defensive_1on1"
    TAKE_ON = "take_on"
    SUBSTITUTION = "substitution"
    BLOCK = "block"
    AERIAL_DUEL = "aerial_duel"
    SHOT = "shot"
    SAVE = "save"
    FOUL = "foul"
    FOUL_WON = "foul_won"
    GOAL = "goal"
    CATCH_INTERCEPTION = "catch_interception"
    END_OF_HALF = "end_of_half"


class ActionConfig(NamedTuple):
    weight: float
    min_score: float
    tolerance_seconds: float


ACTION_CONFIGS: dict[Action, ActionConfig] = {
    Action.PASS: ActionConfig(1.0, 0.0, 1.0),
    Action.PASS_RECEIVED: ActionConfig(1.4, 0.0, 1.0),
    Action.RECOVERY: ActionConfig(1.5, 0.0, 1.5),
    Action.TACKLE: ActionConfig(2.5, 0.1, 1.5),
    Action.INTERCEPTION: ActionConfig(2.8, 0.5, 2.0),
    Action.BALL_OUT_OF_PLAY: ActionConfig(2.9, 0.5, 2.0),
    Action.CLEARANCE: ActionConfig(3.1, 0.5, 2.0),
    Action.DEFENSIVE_1ON1: ActionConfig(3.2, 0.5, 2.0),
    Action.TAKE_ON: ActionConfig(3.2, 0.5, 2.0),
    Action.SUBSTITUTION: ActionConfig(4.2, 0.5, 2.0),
    Action.BLOCK: ActionConfig(4.2, 0.5, 2.0),
    Action.AERIAL_DUEL: ActionConfig(4.3, 0.5, 2.0),
    Action.SHOT: ActionConfig(4.7, 0.5, 2.0),
    Action.SAVE: ActionConfig(7.3, 0.5, 2.0),
    Action.FOUL: ActionConfig(7.7, 0.5, 2.5),
    Action.FOUL_WON: ActionConfig(7.7, 0.5, 2.5),
    Action.GOAL: ActionConfig(10.9, 0.5, 3.0),
    Action.CATCH_INTERCEPTION: ActionConfig(17.2, 0.4, 2.5),
    Action.END_OF_HALF: ActionConfig(17.2, 0.5, 5.0),
}
