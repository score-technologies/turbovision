from unittest.mock import patch
from types import SimpleNamespace
from scorevision.validator.central.private_track.challenges import (
    Challenge,
    has_sufficient_actions,
    select_challenge,
)


_FAKE_SETTINGS = SimpleNamespace(PRIVATE_MIN_ACTIONS_FOR_CHALLENGE=3)


def _patch_settings():
    return patch(
        "scorevision.validator.central.private_track.challenges.get_settings",
        return_value=_FAKE_SETTINGS,
    )


def test_has_sufficient_actions_true():
    with _patch_settings():
        gt = [{"frame": i * 25, "action": "pass"} for i in range(3)]
        assert has_sufficient_actions(gt) is True


def test_has_sufficient_actions_false():
    with _patch_settings():
        gt = [{"frame": 25, "action": "pass"}]
        assert has_sufficient_actions(gt) is False


def test_has_sufficient_actions_empty():
    with _patch_settings():
        assert has_sufficient_actions([]) is False


def test_select_challenge_returns_none_when_empty():
    with _patch_settings():
        assert select_challenge([]) is None


def test_select_challenge_returns_none_when_insufficient_actions():
    with _patch_settings():
        segments = [
            {
                "video_id": "v1",
                "video_url": "https://example.com/v1.mp4",
                "ground_truth": [{"frame": 25, "action": "pass"}],
            }
        ]
        assert select_challenge(segments) is None


def test_select_challenge_returns_challenge():
    with _patch_settings():
        gt = [{"frame": i * 25, "action": "pass"} for i in range(5)]
        segments = [
            {
                "video_id": "v1",
                "video_url": "https://example.com/v1.mp4",
                "ground_truth": gt,
            }
        ]
        challenge = select_challenge(segments)
        assert challenge is not None
        assert isinstance(challenge, Challenge)
        assert challenge.video_url == "https://example.com/v1.mp4"
        assert len(challenge.ground_truth) == 5
        assert challenge.challenge_id.startswith("v1_")
