import asyncio
from types import SimpleNamespace

import pytest

from scorevision.validator.audit.open_source import compliance as compliance_mod
from scorevision.validator.audit.open_source import security as security_mod


def test_security_runner_is_initialized_and_callable():
    assert callable(compliance_mod._get_security_runner())


def test_security_runner_falls_back_when_loader_fails(monkeypatch):
    monkeypatch.setattr(compliance_mod, "_SECURITY_RUNNER", None)

    def boom():
        raise ImportError("missing dependency")

    monkeypatch.setattr(compliance_mod, "_load_security_runner", boom)
    fn = compliance_mod._get_security_runner()
    out = fn(model_repo="a/b", revision="r1", payload_frames=[])
    assert out.success is True
    assert out.latency_ms == 0.0


def test_p95_uses_sorted_index_percentile():
    values = [10.0, 30.0, 20.0, 50.0, 40.0]
    # index = int(0.95 * (n - 1)) = int(3.8) = 3 -> sorted[3] = 40
    assert compliance_mod._p95(values) == 40.0


def test_compare_predictions_iou_success_when_boxes_match():
    expected = {
        "frames": [
            {
                "frame_id": 1,
                "boxes": [
                    {"cls_id": 0, "x1": 10, "y1": 10, "x2": 20, "y2": 20},
                ],
            }
        ]
    }
    actual = {
        "frames": [
            {
                "frame_id": 1,
                "boxes": [
                    {"cls_id": 0, "x1": 10, "y1": 10, "x2": 20, "y2": 20},
                ],
            }
        ]
    }

    ok, info = compliance_mod._compare_predictions_iou(expected, actual, threshold=0.9)

    assert ok is True
    assert info["mean_iou"] == 1.0
    assert info["frames_compared"] == 1
    assert info["extra_boxes"] == 0
    assert info["missing_boxes"] == 0


def test_compare_predictions_iou_success_when_polygons_match():
    expected = {
        "frames": [
            {
                "frame_id": 1,
                "polygons": [
                    {
                        "cls_id": 0,
                        "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                    },
                ],
            }
        ]
    }
    actual = {
        "frames": [
            {
                "frame_id": 1,
                "polygons": [
                    {
                        "cls_id": 0,
                        "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                    },
                ],
            }
        ]
    }

    ok, info = compliance_mod._compare_predictions_iou(expected, actual, threshold=0.9)

    assert ok is True
    assert info["mean_iou"] == 1.0
    assert info["extra_detections"] == 0
    assert info["missing_detections"] == 0


def test_compare_predictions_iou_accepts_box_against_polygon():
    expected = {
        "frames": [
            {
                "frame_id": 1,
                "boxes": [
                    {"cls_id": 0, "x1": 10, "y1": 10, "x2": 20, "y2": 20},
                ],
            }
        ]
    }
    actual = {
        "frames": [
            {
                "frame_id": 1,
                "polygons": [
                    {
                        "cls_id": 0,
                        "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                    },
                ],
            }
        ]
    }

    ok, info = compliance_mod._compare_predictions_iou(expected, actual, threshold=0.9)

    assert ok is True
    assert info["mean_iou"] == 1.0


def test_security_output_validation_accepts_polygon_only_frames():
    security_mod._validate_prediction_output(
        [
            {
                "frame_id": 1,
                "polygons": [
                    {
                        "cls_id": 0,
                        "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                    }
                ],
            }
        ]
    )


def test_compare_predictions_iou_fails_without_common_frames():
    expected = {"frames": [{"frame_id": 1, "boxes": []}]}
    actual = {"frames": [{"frame_id": 2, "boxes": []}]}

    ok, info = compliance_mod._compare_predictions_iou(expected, actual, threshold=0.5)

    assert ok is False
    assert info["reason"] == "no_common_frames"


def test_run_public_compliance_once_fails_when_checker_r2_unconfigured(monkeypatch):
    monkeypatch.setattr(compliance_mod, "checker_r2_config", lambda: object())
    monkeypatch.setattr(compliance_mod, "is_configured", lambda _cfg, require_bucket=True: False)

    with pytest.raises(RuntimeError, match="Checker R2 not configured"):
        asyncio.run(compliance_mod.run_public_compliance_once())


def test_compliance_loop_triggers_run_when_interval_elapsed(monkeypatch):
    calls = {"run": 0, "sleep": 0}

    class FakeSubtensor:
        async def get_current_block(self):
            return 100

    async def fake_get_subtensor():
        return FakeSubtensor()

    async def fake_run_once():
        calls["run"] += 1
        return {"winners_block": 99, "targets": 3}

    async def fake_load_last_trigger_block():
        return 0

    async def fake_sleep(_seconds):
        calls["sleep"] += 1
        raise asyncio.CancelledError()

    settings = SimpleNamespace(
        CHECKER_INTERVAL_BLOCKS=10,
        CHECKER_POLL_INTERVAL_S=0,
    )

    monkeypatch.setattr(compliance_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(compliance_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(compliance_mod, "run_public_compliance_once", fake_run_once)
    monkeypatch.setattr(compliance_mod, "_load_last_trigger_block_from_runs_index", fake_load_last_trigger_block)
    monkeypatch.setattr(compliance_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(compliance_mod.compliance_loop())

    assert calls["run"] == 1
    assert calls["sleep"] == 1
