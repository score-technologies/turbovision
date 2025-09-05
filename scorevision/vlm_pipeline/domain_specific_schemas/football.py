from enum import Enum


class Person(Enum):
    PLAYER = "player"
    REFEREE = "referee"
    BALL = "ball"
    GOALIE = "goalkeeper"


class Action(Enum):
    NONE = "No Special Action"
    PENALTY = "Penalty"
    KICK_OFF = "Kick-off"
    GOAL = "Goal"
    SUB = "Substitution"
    OFFSIDE = "Offside"
    SHOT_ON_TARGET = "Shots on target"
    SHOT_OFF_TARGET = "Shots off target"
    CLEARANCE = "Clearance"
    BALL_OOP = "Ball out of play"
    THROW_IN = "Throw-in"
    FOUL = "Foul"
    INDIRECT_FREE_KICK = "Indirect free-kick"
    DIRECT_FREE_KICK = "Direct free-kick"
    CORNER = "Corner"
    YELLOW_CARD = "Yellow card"
    RED_CARD = "Red card"
    YELLOW_RED_CARD = "Yellow->red card"
