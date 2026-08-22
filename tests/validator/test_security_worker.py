import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from scorevision.validator.audit.open_source import security as sec


MINER_SRC = """
class Miner:
    def __init__(self, path_hf_repo=None):
        self.path = path_hf_repo

    def predict_batch(self, batch_images, offset, n_keypoints):
        return [{"boxes": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "cls_id": 0}], "polygons": []}
                for _ in batch_images]
"""


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "miner.py").write_text(MINER_SRC)
    (repo / "model.onnx").write_bytes(b"\x00" * 16)
    return repo


def _frame() -> dict:
    ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return {"frame_id": 0, "data": base64.b64encode(buf.tobytes()).decode()}


def test_worker_runs_without_downloading_anything(monkeypatch, fake_repo):
    """The parent hands over a local path; the worker itself never fetches."""
    calls = []

    def fake_snapshot_download(repo_id, revision=None, **kwargs):
        calls.append((repo_id, revision))
        return str(fake_repo)

    monkeypatch.setattr(sec, "snapshot_download", fake_snapshot_download)

    out = sec.run_local_inference_from_hf(
        model_repo="acme/model",
        revision="deadbeef",
        payload_frames=[_frame()],
        wall_timeout_seconds=30,
    )

    assert out.success, out.error
    assert out.predictions["frames"][0]["frame_id"] == 0
    # downloaded exactly once, in the parent process
    assert calls == [("acme/model", "deadbeef")]


def test_repo_is_vetted_before_the_worker_starts(monkeypatch, tmp_path):
    """A repo without an onnx model is rejected without spawning the worker."""
    repo = tmp_path / "bad"
    repo.mkdir()
    (repo / "miner.py").write_text(MINER_SRC)
    monkeypatch.setattr(sec, "snapshot_download", lambda *a, **k: str(repo))

    worker = sec.PersistentInferenceWorker(model_repo="acme/model", revision="rev")
    with pytest.raises(ValueError, match="no_onnx_model_found"):
        worker.start()
    assert worker._proc is None


def test_harden_worker_process_removes_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKER_R2_WRITE_SECRET_ACCESS_KEY", "leak-me")
    sec._harden_worker_process(str(tmp_path))
    assert "CHECKER_R2_WRITE_SECRET_ACCESS_KEY" not in __import__("os").environ


def test_symlink_out_of_the_repo_is_rejected(tmp_path):
    """A repo linking to the parent's filesystem is refused before anything reads it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "miner.py").write_text(MINER_SRC)
    (repo / "model.onnx").write_bytes(b"\x00" * 16)
    secret = tmp_path / "wallet.json"
    secret.write_text("hotkey")
    (repo / "notes.txt").symlink_to(secret)

    with pytest.raises(ValueError, match="symlink_escapes_repo"):
        sec._validate_repo_artifacts(repo, max_repo_bytes=10_000_000)


def test_oversized_miner_py_is_rejected_before_parsing(tmp_path):
    miner_py = tmp_path / "miner.py"
    miner_py.write_text("x = 1\n" * 200_000)

    with pytest.raises(ValueError, match="miner_py_too_large"):
        sec._scan_miner_ast(miner_py)


def test_hf_token_is_passed_explicitly_not_read_from_env(monkeypatch, fake_repo):
    """The env is scrubbed before miner code runs, so the token must be passed in."""
    seen = {}

    def fake_snapshot_download(repo_id, revision=None, token=None, **kwargs):
        seen["token"] = token
        return str(fake_repo)

    monkeypatch.setattr(sec, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(sec, "_hf_token", lambda: "hf_from_settings")

    worker = sec.PersistentInferenceWorker(model_repo="acme/model", revision="rev")
    try:
        worker.start()
    finally:
        worker.close()

    assert seen["token"] == "hf_from_settings"


def test_repo_is_staged_readable_and_cleaned_up(monkeypatch, fake_repo):
    """The worker reads a copy it can still access after dropping privileges."""
    monkeypatch.setattr(sec, "snapshot_download", lambda *a, **k: str(fake_repo))

    worker = sec.PersistentInferenceWorker(model_repo="acme/model", revision="rev")
    staged = worker._prepare_repo()
    try:
        assert staged != fake_repo
        assert (staged / "miner.py").read_text() == MINER_SRC
        assert (staged / "miner.py").stat().st_mode & 0o044  # world readable
        assert staged.stat().st_mode & 0o055  # world traversable
    finally:
        worker.close()
    assert not staged.exists()


def test_drop_privileges_warns_instead_of_failing_when_not_root(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        sec._drop_privileges(str(tmp_path))
    assert "cannot drop privileges" in caplog.text


def test_worker_reports_its_privilege_state_to_the_parent(monkeypatch, fake_repo, caplog):
    """Worker logging is muted, so the drop status has to travel back on the pipe."""
    monkeypatch.setattr(sec, "snapshot_download", lambda *a, **k: str(fake_repo))

    worker = sec.PersistentInferenceWorker(model_repo="acme/model", revision="rev")
    with caplog.at_level("INFO"):
        try:
            worker.start()
        finally:
            worker.close()

    assert "privileges=" in caplog.text
    assert "uid=" in caplog.text
