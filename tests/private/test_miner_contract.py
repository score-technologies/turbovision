from scorevision.miner.open_source.chute_template.schemas import SVFrameResult, SVBox, SVPolygon


def test_sv_frame_result_accepts_optional_polygons():
    payload = {
        "frame_id": 1,
        "boxes": [
            {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "cls_id": 0, "conf": 0.9}
        ],
        "polygons": [
            {"cls_id": 0, "conf": 0.8, "points": [(1, 2), (3, 2), (3, 4), (1, 4)]}
        ],
        "keypoints": [(5, 6)],
    }

    result = SVFrameResult(**payload)

    assert result.frame_id == 1
    assert isinstance(result.boxes[0], SVBox)
    assert isinstance(result.polygons[0], SVPolygon)
    assert result.polygons[0].points == [(1, 2), (3, 2), (3, 4), (1, 4)]


def test_sv_frame_result_still_accepts_bbox_only_payloads():
    result = SVFrameResult(
        frame_id=2,
        boxes=[SVBox(x1=1, y1=2, x2=3, y2=4, cls_id=0, conf=0.9)],
        keypoints=[(5, 6)],
    )

    assert result.polygons is None
    assert result.boxes[0].cls_id == 0
