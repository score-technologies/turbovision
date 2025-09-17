from enum import Enum


class Person(Enum):
    PLAYER = "player"
    REFEREE = "referee"
    BALL = "ball"


class Action(Enum):
    NONE = "No Special Action"
    TIP_OFF = "Tip-off"
    FIELD_GOAL_MADE = "Field goal made"
    FIELD_GOAL_MISSED = "Field goal missed"
    THREE_POINTER_MADE = "Three-pointer made"
    THREE_POINTER_MISSED = "Three-pointer missed"
    FREE_THROW_MADE = "Free throw made"
    FREE_THROW_MISSED = "Free throw missed"
    FOUL = "Foul"
    PERSONAL_FOUL = "Personal foul"
    TECHNICAL_FOUL = "Technical foul"
    FLAGRANT_FOUL = "Flagrant foul"
    BLOCK = "Block"
    STEAL = "Steal"
    TURNOVER = "Turnover"
    ASSIST = "Assist"
    REBOUND_OFFENSIVE = "Offensive rebound"
    REBOUND_DEFENSIVE = "Defensive rebound"
    SUBSTITUTION = "Substitution"
    TIMEOUT = "Timeout"
    JUMP_BALL = "Jump ball"
    VIOLATION = "Violation"
    OUT_OF_BOUNDS = "Out of bounds"
    SHOT_CLOCK_VIOLATION = "Shot clock violation"
    BACKCOURT_VIOLATION = "Backcourt violation"
    TRAVELING = "Traveling"
    DOUBLE_DRIBBLE = "Double dribble"
    GOALTENDING = "Goaltending"
