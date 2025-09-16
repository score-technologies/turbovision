from enum import Enum


class ChallengeType(Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    BASKETBALL = "basketball"


CHALLENGE_ID_LOOKUP = dict(enumerate(ChallengeType))
