from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import logging
import multiprocessing as mp
import os
import resource
import shutil
import socket
import sys
import tempfile
from base64 import b64decode
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.request import Request, urlopen

from cv2 import IMREAD_COLOR, imdecode
from huggingface_hub import snapshot_download
from numpy import frombuffer, uint8


logger = logging.getLogger("scorevision.security")


DISALLOWED_IMPORT_ROOTS = {
    "socket",
    "subprocess",
    "ctypes",
    "multiprocessing",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "paramiko",
}
DISALLOWED_CALL_NAMES = {
    "eval",
    "exec",
    "__import__",
    "open",
}
DISALLOWED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("subprocess", "Popen"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_output"),
}
ALLOWED_SUFFIXES = {
    ".onnx",
    ".py",
    ".json",
    ".txt",
    ".md",
    ".yml",
    ".yaml",
    ".labels",
    ".csv",
}
DISALLOWED_BINARY_SUFFIXES = {
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
}


@dataclass
class LocalRunResult:
    success: bool
    predictions: dict[str, Any] | None
    latency_ms: float
    error: str | None = None
    memory_mb_peak: float | None = None


def _load_miner_from_hf_repo(
    *,
    path_hf_repo: Path,
    filename: str = "miner.py",
    classname: str = "Miner",
):
    module_path = Path(path_hf_repo) / filename
    if not module_path.exists():
        raise ValueError(f"missing_miner_file:{module_path}")

    spec = importlib.util.spec_from_file_location("scorevision_hf_miner", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"invalid_miner_spec:{module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls = getattr(mod, classname, None)
    if cls is None:
        raise ValueError(f"missing_miner_class:{classname}")

    try:
        return cls(path_hf_repo=path_hf_repo)
    except TypeError:
        try:
            return cls(path_hf_repo=str(path_hf_repo))
        except TypeError:
            return cls()


def _scan_miner_ast(miner_path: Path) -> None:
    source = miner_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(miner_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"disallowed import '{root}' in miner.py")
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"disallowed import-from '{root}' in miner.py")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALL_NAMES:
                raise ValueError(f"disallowed call '{node.func.id}' in miner.py")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in DISALLOWED_ATTR_CALLS:
                    raise ValueError(f"disallowed call '{pair[0]}.{pair[1]}' in miner.py")


def _validate_repo_artifacts(repo_path: Path, max_repo_bytes: int) -> tuple[int, int]:
    total_bytes = 0
    onnx_count = 0
    for p in repo_path.rglob("*"):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix.lower() == ".pyc":
            continue

        total_bytes += p.stat().st_size
        suffix = p.suffix.lower()
        if suffix == ".onnx":
            onnx_count += 1
        if suffix in DISALLOWED_BINARY_SUFFIXES:
            raise ValueError(f"disallowed_binary_artifact:{p.name}")
        if suffix and suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"unsupported_artifact_suffix:{p.name}")

    if total_bytes > max_repo_bytes:
        raise ValueError(f"repo_size_exceeded:{total_bytes}>{max_repo_bytes}")
    if onnx_count == 0:
        raise ValueError("no_onnx_model_found")
    return total_bytes, onnx_count


def _decode_payload_frames(frames: list[dict[str, Any]]) -> tuple[list[Any], list[int]]:
    decoded = []
    frame_ids: list[int] = []

    for item in frames:
        frame_id = int(item.get("frame_id", 0))

        raw = item.get("data")
        if raw:
            content = raw if isinstance(raw, str) else str(raw)
            arr = frombuffer(b64decode(content), dtype=uint8)
            img = imdecode(arr, IMREAD_COLOR)
            if img is None:
                raise ValueError(f"failed_to_decode_base64_frame_id={frame_id}")
            decoded.append(img)
            frame_ids.append(frame_id)
            continue

        url = item.get("url")
        if isinstance(url, str) and url:
            req = Request(url, headers={"User-Agent": "scorevision-compliance/1.0"})
            with urlopen(req, timeout=15) as resp:
                blob = resp.read()
            arr = frombuffer(blob, dtype=uint8)
            img = imdecode(arr, IMREAD_COLOR)
            if img is None:
                raise ValueError(f"failed_to_decode_url_frame_id={frame_id}")
            decoded.append(img)
            frame_ids.append(frame_id)
            continue

        raise ValueError("payload frame missing base64 data/url")

    return decoded, frame_ids


def _validate_miner_interface(miner: Any) -> None:
    predict_batch = getattr(miner, "predict_batch", None)
    if not callable(predict_batch):
        raise ValueError("miner_missing_predict_batch")
    sig = inspect.signature(predict_batch)
    required = {"batch_images", "offset", "n_keypoints"}
    if not required.issubset(set(sig.parameters.keys())):
        raise ValueError("predict_batch_signature_invalid")


def _validate_prediction_output(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        if "frame_id" not in row:
            raise ValueError(f"output_missing_frame_id_at_{idx}")
        boxes = row.get("boxes") or []
        polygons = row.get("polygons") or []
        if not isinstance(boxes, list):
            raise ValueError(f"output_boxes_invalid_at_{idx}")
        if not isinstance(polygons, list):
            raise ValueError(f"output_polygons_invalid_at_{idx}")
        for j, box in enumerate(boxes):
            for key in ("x1", "y1", "x2", "y2", "cls_id"):
                if key not in box:
                    raise ValueError(f"output_box_missing_{key}_at_{idx}_{j}")
        for j, polygon in enumerate(polygons):
            if "cls_id" not in polygon:
                raise ValueError(f"output_polygon_missing_cls_id_at_{idx}_{j}")
            points = polygon.get("points") or polygon.get("polygon") or polygon.get("masks")
            if not isinstance(points, list) or not points:
                raise ValueError(f"output_polygon_points_invalid_at_{idx}_{j}")
            for k, point in enumerate(points):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"output_polygon_point_invalid_at_{idx}_{j}_{k}")


def _safe_setrlimit(which: int, soft: int, hard: int | None = None) -> None:
    try:
        _cur_soft, cur_hard = resource.getrlimit(which)
        max_hard = cur_hard if cur_hard != resource.RLIM_INFINITY else soft
        target_soft = min(soft, max_hard)
        target_hard = target_soft if hard is None else min(hard, max_hard)
        resource.setrlimit(which, (target_soft, target_hard))
    except Exception:
        pass


def _peak_rss_mb() -> float:
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak_rss / (1024.0 * 1024.0)
    return peak_rss / 1024.0


def _sandbox_limits(*, memory_bytes: int, max_processes: int) -> None:
    # Keep memory/process/file protections.
    # CPU hard limit disabled for persistent worker mode to avoid SIGXCPU (-24)
    # after accumulated inferences.
    _safe_setrlimit(resource.RLIMIT_AS, memory_bytes)
    _safe_setrlimit(resource.RLIMIT_NPROC, max_processes)
    _safe_setrlimit(resource.RLIMIT_NOFILE, 128)


def _worker_main(conn, *, memory_bytes: int, cpu_seconds: int):
    _ = cpu_seconds  # kept for API compatibility
    tmp_dir = tempfile.mkdtemp(prefix="sv-comp-")
    miner = None
    try:
        logging.raiseExceptions = False
        logging.disable(logging.CRITICAL)
        _sandbox_limits(memory_bytes=memory_bytes, max_processes=16)
        os.chdir(tmp_dir)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        while True:
            msg = conn.recv()
            op = msg.get("op")

            if op == "close":
                conn.send({"ok": True})
                return

            if op == "init":
                model_repo = str(msg["model_repo"])
                revision = str(msg["revision"])
                max_repo_bytes = int(msg["max_repo_bytes"])
                t0 = monotonic()

                repo_path = Path(snapshot_download(model_repo, revision=revision))
                miner_py = repo_path / "miner.py"
                if not miner_py.exists():
                    conn.send({"ok": False, "error": "miner.py missing in HF repo"})
                    continue

                _validate_repo_artifacts(repo_path, max_repo_bytes=max_repo_bytes)
                _scan_miner_ast(miner_py)
                miner = _load_miner_from_hf_repo(path_hf_repo=repo_path, filename="miner.py", classname="Miner")
                _validate_miner_interface(miner)

                conn.send(
                    {
                        "ok": True,
                        "init_ms": (monotonic() - t0) * 1000.0,
                        "model_repo": model_repo,
                        "revision": revision,
                    }
                )
                continue

            if op == "infer":
                if miner is None:
                    conn.send({"ok": False, "error": "worker_not_initialized"})
                    continue

                payload_frames = msg["payload_frames"]
                n_keypoints = int(msg["n_keypoints"])
                challenge_id = str(msg.get("challenge_id", "unknown"))

                try:
                    t_decode0 = monotonic()
                    images, frame_ids = _decode_payload_frames(payload_frames)
                    decode_ms = (monotonic() - t_decode0) * 1000.0

                    def _deny_socket(*args, **kwargs):
                        raise RuntimeError("network_disabled_in_compliance_runner")

                    socket.socket = _deny_socket  # type: ignore[assignment]

                    t_infer0 = monotonic()
                    frame_results = miner.predict_batch(
                        batch_images=images,
                        offset=0,
                        n_keypoints=n_keypoints,
                    )
                    infer_ms = (monotonic() - t_infer0) * 1000.0
                    memory_mb_peak = _peak_rss_mb()

                    rows = []
                    for frame_id, frame_result in zip(frame_ids, frame_results, strict=True):
                        dumped = frame_result.model_dump() if hasattr(frame_result, "model_dump") else dict(frame_result)
                        dumped["frame_id"] = frame_id
                        rows.append(dumped)

                    _validate_prediction_output(rows)

                    conn.send(
                        {
                            "ok": True,
                            "challenge_id": challenge_id,
                            "decode_ms": decode_ms,
                            "infer_ms": infer_ms,
                            "latency_ms": infer_ms,
                            "memory_mb_peak": memory_mb_peak,
                            "predictions": {"frames": rows},
                        }
                    )
                except Exception as e:
                    conn.send({"ok": False, "error": f"worker_infer_error:{type(e).__name__}:{e}"})
                finally:
                    try:
                        importlib.reload(socket)
                    except Exception:
                        pass
                continue

            conn.send({"ok": False, "error": f"unknown_op:{op}"})

    except Exception as e:
        try:
            conn.send({"ok": False, "error": f"worker_fatal:{type(e).__name__}:{e}"})
        except Exception:
            pass
    finally:
        try:
            logging.shutdown()
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


class PersistentInferenceWorker:
    def __init__(
        self,
        *,
        model_repo: str,
        revision: str,
        n_keypoints: int = 32,
        max_repo_bytes: int = 30 * 1024 * 1024,
        memory_bytes: int = 8 * 1024 * 1024 * 1024,
        cpu_seconds: int = 30,
        wall_timeout_seconds: int = 45,
    ):
        self.model_repo = model_repo
        self.revision = revision
        self.n_keypoints = n_keypoints
        self.max_repo_bytes = max_repo_bytes
        self.memory_bytes = memory_bytes
        self.cpu_seconds = cpu_seconds
        self.wall_timeout_seconds = wall_timeout_seconds

        self._ctx = mp.get_context("spawn")
        self._parent_conn = None
        self._proc = None

    def start(self) -> None:
        if self._proc is not None:
            return

        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn,),
            kwargs={
                "memory_bytes": self.memory_bytes,
                "cpu_seconds": self.cpu_seconds,
            },
            daemon=False,
        )
        proc.start()
        child_conn.close()

        self._parent_conn = parent_conn
        self._proc = proc

        logger.info(
            "[worker] start model=%s revision=%s",
            self.model_repo,
            self.revision,
        )

        t0 = monotonic()
        self._parent_conn.send(
            {
                "op": "init",
                "model_repo": self.model_repo,
                "revision": self.revision,
                "max_repo_bytes": self.max_repo_bytes,
            }
        )
        if not self._parent_conn.poll(self.wall_timeout_seconds):
            self.close()
            raise RuntimeError("worker_init_timeout")
        out = self._parent_conn.recv()
        if not out.get("ok"):
            self.close()
            raise RuntimeError(out.get("error", "worker_init_failed"))

        logger.info(
            "[worker] ready model=%s revision=%s init_ms=%.1f total_start_ms=%.1f",
            self.model_repo,
            self.revision,
            float(out.get("init_ms", 0.0)),
            (monotonic() - t0) * 1000.0,
        )

    def infer(self, *, payload_frames: list[dict[str, Any]], challenge_id: str) -> LocalRunResult:
        if self._proc is None or self._parent_conn is None:
            return LocalRunResult(False, None, 0.0, "worker_not_started")

        try:
            self._parent_conn.send(
                {
                    "op": "infer",
                    "payload_frames": payload_frames,
                    "n_keypoints": self.n_keypoints,
                    "challenge_id": challenge_id,
                }
            )
        except BrokenPipeError:
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_broken_pipe_on_send:exitcode={exitcode}")
        except Exception as e:
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_send_exception:{type(e).__name__}:{e}:exitcode={exitcode}")

        if not self._parent_conn.poll(self.wall_timeout_seconds):
            logger.warning("[worker] infer timeout challenge_id=%s", challenge_id)
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_infer_timeout:exitcode={exitcode}")

        try:
            out = self._parent_conn.recv()
        except EOFError:
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_channel_eof:exitcode={exitcode}")
        except BrokenPipeError:
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_broken_pipe_on_recv:exitcode={exitcode}")
        except Exception as e:
            exitcode = self._proc.exitcode if self._proc is not None else None
            return LocalRunResult(False, None, 0.0, f"worker_recv_exception:{type(e).__name__}:{e}:exitcode={exitcode}")

        if not out.get("ok"):
            err = out.get("error", "worker_infer_failed")
            logger.warning("[worker] infer failed challenge_id=%s err=%s", challenge_id, err)
            return LocalRunResult(False, None, 0.0, str(err))

        decode_ms = float(out.get("decode_ms", 0.0))
        infer_ms = float(out.get("infer_ms", 0.0))
        memory_mb_peak = float(out.get("memory_mb_peak", 0.0))
        logger.info(
            "[worker] infer done challenge_id=%s decode_ms=%.1f infer_ms=%.1f memory_mb_peak=%.1f",
            challenge_id,
            decode_ms,
            infer_ms,
            memory_mb_peak,
        )
        return LocalRunResult(
            True,
            out.get("predictions"),
            float(out.get("latency_ms", 0.0)),
            None,
            memory_mb_peak,
        )

    def close(self) -> None:
        if self._proc is None or self._parent_conn is None:
            return

        try:
            self._parent_conn.send({"op": "close"})
            self._parent_conn.poll(1.0)
        except Exception:
            pass

        try:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        except Exception:
            pass

        try:
            self._parent_conn.close()
        except Exception:
            pass

        logger.info("[worker] close model=%s revision=%s", self.model_repo, self.revision)

        self._proc = None
        self._parent_conn = None


def run_local_inference_from_hf(
    *,
    model_repo: str,
    revision: str,
    payload_frames: list[dict[str, Any]],
    n_keypoints: int = 32,
    max_repo_bytes: int = 30 * 1024 * 1024,
    memory_bytes: int = 8 * 1024 * 1024 * 1024,
    cpu_seconds: int = 30,
    wall_timeout_seconds: int = 45,
) -> LocalRunResult:
    worker = PersistentInferenceWorker(
        model_repo=model_repo,
        revision=revision,
        n_keypoints=n_keypoints,
        max_repo_bytes=max_repo_bytes,
        memory_bytes=memory_bytes,
        cpu_seconds=cpu_seconds,
        wall_timeout_seconds=wall_timeout_seconds,
    )
    try:
        worker.start()
        return worker.infer(payload_frames=payload_frames, challenge_id="oneshot")
    except Exception as e:
        return LocalRunResult(False, None, 0.0, f"runtime_failed:{type(e).__name__}:{e}")
    finally:
        worker.close()
