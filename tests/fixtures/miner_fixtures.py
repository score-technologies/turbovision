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
    return dict(
        label="player",
        score=1.0,
        cluster_id=1,
        geometry={
            "type": "bbox",
            "points": [{"x": 10, "y": 23}, {"x": 100, "y": 200}],
        },
    )


@fixture
def fake_bboxes(fake_bbox) -> list[dict]:
    return [fake_bbox, fake_bbox, fake_bbox, fake_bbox, fake_bbox]


@fixture
def fake_frame_results(fake_bboxes, fake_keypoints) -> list[dict]:
    return [
        dict(frame_id=1, annotations=fake_bboxes, keypoints=fake_keypoints),
        dict(frame_id=2, annotations=fake_bboxes, keypoints=fake_keypoints),
        dict(frame_id=3, annotations=fake_bboxes, keypoints=fake_keypoints),
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
