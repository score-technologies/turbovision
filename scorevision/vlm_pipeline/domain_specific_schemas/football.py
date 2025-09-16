from enum import Enum

from numpy import ndarray
from cv2 import imread

FOOTBALL_KEYPOINTS: list[tuple[int, int]] = [
    (5, 5),  # 1
    (5, 140),  # 2
    (5, 250),  # 3
    (5, 430),  # 4
    (5, 540),  # 5
    (5, 675),  # 6
    # -------------
    (55, 250),  # 7
    (55, 430),  # 8
    # -------------
    (110, 340),  # 9
    # -------------
    (165, 140),  # 10
    (165, 270),  # 11
    (165, 410),  # 12
    (165, 540),  # 13
    # -------------
    (527, 5),  # 14
    (527, 253),  # 15
    (527, 433),  # 16
    (527, 675),  # 17
    # -------------
    (888, 140),  # 18
    (888, 270),  # 19
    (888, 410),  # 20
    (888, 540),  # 21
    # -------------
    (940, 340),  # 22
    # -------------
    (998, 250),  # 23
    (998, 430),  # 24
    # -------------
    (1045, 5),  # 25
    (1045, 140),  # 26
    (1045, 250),  # 27
    (1045, 430),  # 28
    (1045, 540),  # 29
    (1045, 675),  # 30
    # -------------
    (435, 340),  # 31
    (615, 340),  # 32
]

INDEX_KEYPOINT_CORNER_BOTTOM_LEFT = 5
INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT = 29
INDEX_KEYPOINT_CORNER_TOP_LEFT = 0
INDEX_KEYPOINT_CORNER_TOP_RIGHT = 24


def football_pitch() -> ndarray:
    return imread(
        "scorevision/vlm_pipeline/domain_specific_schemas/football_pitch_template.png"
    )


class Person(Enum):
    BALL = "ball"
    GOALIE = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"


OBJECT_ID_LOOKUP = {
    0: Person.BALL,
    1: Person.GOALIE,
    2: Person.PLAYER,
    3: Person.REFEREE,
    6: "team 1",
    7: "team 2",
}


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
