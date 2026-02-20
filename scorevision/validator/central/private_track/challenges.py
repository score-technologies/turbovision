import random
from dataclasses import dataclass
from scorevision.utils.schemas import FramePrediction
from scorevision.utils.settings import get_settings


@dataclass
class Challenge:
    challenge_id: str
    video_url: str
    ground_truth: list[FramePrediction]


def has_sufficient_actions(ground_truth: list) -> bool:
    return len(ground_truth) >= get_settings().PRIVATE_MIN_ACTIONS_FOR_CHALLENGE


def select_challenge(segments: list[dict]) -> Challenge | None:
    valid = [s for s in segments if has_sufficient_actions(s.get("ground_truth", []))]
    if not valid:
        return None

    segment = random.choice(valid)
    ground_truth = [
        FramePrediction(frame=gt["frame"], action=gt["action"])
        for gt in segment["ground_truth"]
    ]
    return Challenge(
        challenge_id=f"{segment['video_id']}_{random.randint(1000, 9999)}",
        video_url=segment["video_url"],
        ground_truth=ground_truth,
    )
