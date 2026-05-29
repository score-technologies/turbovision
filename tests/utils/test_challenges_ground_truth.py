from scorevision.utils.challenges import _parse_ground_truth_payload


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
