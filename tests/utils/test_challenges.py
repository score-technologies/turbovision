from scorevision.utils.challenges import _parse_ground_truth_payload


def test_parse_ground_truth_payload_accepts_bbox_ground_truth():
    payload = {
        "annotations": [
            {
                "type": "bbox",
                "bbox": [10, 20, 30, 40],
                "label": "player",
                "frame_idx": 2,
            }
        ]
    }

    parsed = _parse_ground_truth_payload(payload, challenge_id=123)

    assert len(parsed) == 1
    assert parsed[0].frame_number == 2
    assert len(parsed[0].annotation.bboxes) == 1
    assert parsed[0].annotation.bboxes[0].label == "player"
    assert parsed[0].annotation.bboxes[0].geometry.type.value == "bbox"


def test_parse_ground_truth_payload_skips_malformed_polygon_annotation():
    payload = {
        "annotations": [
            {
                "type": "polygon",
                "polygon": [
                    {"x": 10, "y": 20},
                    {"x": "bad", "y": 30},
                    {"x": 30, "y": 40},
                ],
                "label": "player",
                "frame_idx": 2,
            }
        ]
    }

    parsed = _parse_ground_truth_payload(payload, challenge_id=123)

    assert parsed == []
