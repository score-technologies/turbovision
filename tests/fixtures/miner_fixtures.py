from pytest import fixture

from scorevision.utils.data_models import SVRunOutput


@fixture
def fake_keypoints() -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]


@fixture
def fake_bbox() -> dict:
    return dict(x1=10, y1=23, x2=100, y2=200, cls_id=1, conf=1.0)


@fixture
def fake_bboxes(fake_bbox) -> list[dict]:
    return [fake_bbox, fake_bbox, fake_bbox, fake_bbox, fake_bbox]


@fixture
def fake_frame_results(fake_bboxes, fake_keypoints) -> list[dict]:
    return [
        dict(frame_id=1, boxes=fake_bboxes, keypoints=fake_keypoints),
        dict(frame_id=2, boxes=fake_bboxes, keypoints=fake_keypoints),
        dict(frame_id=3, boxes=fake_bboxes, keypoints=fake_keypoints),
    ]


@fixture
def fake_miner_predictions(fake_frame_results) -> SVRunOutput:
    return SVRunOutput(
        success=True,
        latency_ms=0.0,
        predictions=dict(frames=fake_frame_results),
        error=None,
        model=None,
    )
