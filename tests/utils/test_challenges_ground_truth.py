import pytest

from scorevision.utils.challenges import (
    ScoreVisionChallengeError,
    _parse_ground_truth_payload,
)


def test_parse_ground_truth_payload_preserves_polygon():
    payload = {
        "annotations": [
            {
                "frame_idx": 3,
                "bbox": [10, 20, 30, 40],
                "polygon": [[10, 20], [30, 20], [30, 40], [10, 40]],
                "label": "player",
            }
        ]
    }

    parsed = _parse_ground_truth_payload(payload, challenge_id=123)

    assert len(parsed) == 1
    box = parsed[0].annotation.bboxes[0]
    assert box.bbox_2d == (10, 20, 30, 40)
    assert box.polygon == [(10, 20), (30, 20), (30, 40), (10, 40)]


def test_parse_ground_truth_payload_keeps_bbox_only_payloads():
    payload = {
        "annotations": [
            {
                "frame_idx": 1,
                "bbox": [1, 2, 3, 4],
                "label": "ball",
            }
        ]
    }

    parsed = _parse_ground_truth_payload(payload, challenge_id=123)

    assert len(parsed) == 1
    box = parsed[0].annotation.bboxes[0]
    assert box.bbox_2d == (1, 2, 3, 4)
    assert box.polygon is None


def test_parse_empty_annotations_as_negative_detection_frame():
    parsed = _parse_ground_truth_payload(
        {
            "annotations": [],
            "frame_index": 573,
            "generation": {"no_target_objects": True},
        },
        challenge_id=91969,
    )

    assert len(parsed) == 1
    assert parsed[0].frame_number == 0
    assert parsed[0].annotation.bboxes == []
    assert parsed[0].annotation.reason == "ScoreVision API negative ground truth"


def test_rejects_empty_annotations_without_explicit_negative_marker():
    with pytest.raises(
        ScoreVisionChallengeError,
        match="not marked as an explicit negative sample",
    ):
        _parse_ground_truth_payload(
            {"annotations": [], "generation": {}},
            challenge_id=91970,
        )
