from enum import Enum


class ChallengeType(Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    BASKETBALL = "basketball"


CHALLENGE_ID_LOOKUP = dict(enumerate(ChallengeType))

_STR_TO_TYPE = {
    "football": ChallengeType.FOOTBALL,
    "soccer": ChallengeType.FOOTBALL,
    "cricket": ChallengeType.CRICKET,
    "basketball": ChallengeType.BASKETBALL,
}


def parse_challenge_type(value) -> ChallengeType | None:
    """ """
    if isinstance(value, ChallengeType):
        return value
    if isinstance(value, str):
        return _STR_TO_TYPE.get(value.strip().lower())
    return None
