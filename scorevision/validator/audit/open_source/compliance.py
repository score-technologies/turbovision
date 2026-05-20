from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import random
import threading
from json import dumps
from logging import getLogger
from time import time
from typing import Any
from types import SimpleNamespace
import os

from scorevision.utils.bittensor_helpers import get_subtensor
from scorevision.utils.r2 import (
    add_index_key_if_new,
    create_s3_client,
    ensure_index_exists,
    fetch_index_keys,
    fetch_json_from_url,
    fetch_responses_data,
    fetch_shard_lines,
    is_configured,
    R2Config,
)
from scorevision.utils.r2_public import extract_base_url
from scorevision.utils.settings import get_settings
from scorevision.validator.central.scheduling import load_manifest

logger = getLogger(__name__)

_PREV_THREAD_EXCEPTHOOK = threading.excepthook


def _quiet_thread_eof(args: threading.ExceptHookArgs):
    # Suppress noisy QueueListener EOFError monitor thread tracebacks.
    if args.exc_type is EOFError and args.thread and args.thread.name == "Thread-2":
        return
    _PREV_THREAD_EXCEPTHOOK(args)


threading.excepthook = _quiet_thread_eof


def _load_security_runner():
    module_name = (os.getenv("CHECKER_SECURITY_MODULE") or "").strip()
    file_path = (os.getenv("CHECKER_SECURITY_FILE") or "").strip()
    if module_name:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, "run_local_inference_from_hf", None)
        if callable(fn):
            return fn
        raise RuntimeError(f"module '{module_name}' has no run_local_inference_from_hf")
    if file_path:
        spec = importlib.util.spec_from_file_location("checker_security_local", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load security file: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "run_local_inference_from_hf", None)
        if callable(fn):
            return fn
        raise RuntimeError(f"security file '{file_path}' missing run_local_inference_from_hf")
    mod = importlib.import_module("scorevision.validator.audit.open_source.security")
    fn = getattr(mod, "run_local_inference_from_hf", None)
    if callable(fn):
        return fn
    raise RuntimeError("default security runner unavailable")


def _load_security_module():
    module_name = (os.getenv("CHECKER_SECURITY_MODULE") or "").strip()
    file_path = (os.getenv("CHECKER_SECURITY_FILE") or "").strip()
    if module_name:
        return importlib.import_module(module_name)
    if file_path:
        spec = importlib.util.spec_from_file_location("checker_security_local", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load security file: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module("scorevision.validator.audit.open_source.security")


def _fallback_security_runner(
    *,
    model_repo: str,
    revision: str,
    payload_frames: list[dict[str, Any]],
    n_keypoints: int = 32,
    max_repo_bytes: int = 30 * 1024 * 1024,
    memory_bytes: int = 8 * 1024 * 1024 * 1024,
    cpu_seconds: int = 30,
    wall_timeout_seconds: int = 45,
):
    _ = (
        model_repo,
        revision,
        payload_frames,
        n_keypoints,
        max_repo_bytes,
        memory_bytes,
        cpu_seconds,
        wall_timeout_seconds,
    )
    return SimpleNamespace(
        success=True,
        predictions={"frames": []},
        latency_ms=0.0,
        error=None,
    )


_SECURITY_RUNNER = None
_PERSISTENT_WORKER_CLASS = None


def _get_security_runner():
    global _SECURITY_RUNNER
    if _SECURITY_RUNNER is not None:
        return _SECURITY_RUNNER
    try:
        _SECURITY_RUNNER = _load_security_runner()
    except Exception as e:
        logger.warning("Falling back to stub security runner: %s", e)
        _SECURITY_RUNNER = _fallback_security_runner
    return _SECURITY_RUNNER


def _get_persistent_worker_class():
    global _PERSISTENT_WORKER_CLASS
    if _PERSISTENT_WORKER_CLASS is not None:
        return _PERSISTENT_WORKER_CLASS
    try:
        mod = _load_security_module()
        klass = getattr(mod, "PersistentInferenceWorker", None)
        if klass is not None:
            logger.info("[compliance] using PersistentInferenceWorker from security module")
        _PERSISTENT_WORKER_CLASS = klass
    except Exception as e:
        logger.warning("[compliance] persistent worker unavailable: %s", e)
        _PERSISTENT_WORKER_CLASS = None
    return _PERSISTENT_WORKER_CLASS


def checker_r2_config() -> R2Config:
    s = get_settings()
    return R2Config(
        bucket=(s.CHECKER_R2_BUCKET or "").strip(),
        account_id=s.CHECKER_R2_ACCOUNT_ID.get_secret_value(),
        access_key_id=s.CHECKER_R2_WRITE_ACCESS_KEY_ID.get_secret_value(),
        secret_access_key=s.CHECKER_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value(),
        concurrency=s.CHECKER_R2_CONCURRENCY,
    )


def _checker_prefix() -> str:
    return f"{get_settings().CHECKER_R2_RESULTS_PREFIX.strip().strip('/')}/"


def _checker_runs_index_key() -> str:
    return f"{_checker_prefix()}runs/index.json"


def _checker_runs_key(block: int) -> str:
    return f"{_checker_prefix()}runs/{max(0, int(block)):09d}.json"


def _checker_fails_key() -> str:
    custom = (get_settings().CHECKER_R2_FAILS_KEY or "").strip()
    if custom:
        return custom
    return f"{_checker_prefix()}failing_tuples.json"


def _get_checker_client():
    cfg = checker_r2_config()
    return create_s3_client(cfg, error_message="Checker R2 credentials not set")


async def _put_json(key: str, payload: dict | list) -> None:
    cfg = checker_r2_config()
    async with _get_checker_client() as c:
        await c.put_object(
            Bucket=cfg.bucket,
            Key=key,
            Body=dumps(payload, separators=(",", ":")),
            ContentType="application/json",
        )


async def _get_json(key: str) -> dict | list | None:
    cfg = checker_r2_config()
    async with _get_checker_client() as c:
        try:
            obj = await c.get_object(Bucket=cfg.bucket, Key=key)
            body = await obj["Body"].read()
            return __import__("json").loads(body.decode())
        except Exception:
            return None


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _compare_predictions_iou(
    expected: dict[str, Any],
    actual: dict[str, Any],
    threshold: float,
) -> tuple[bool, dict[str, Any]]:
    exp_frames = {int(f.get("frame_id", -1)): f for f in (expected.get("frames") or [])}
    act_frames = {int(f.get("frame_id", -1)): f for f in (actual.get("frames") or [])}
    common_ids = sorted(set(exp_frames.keys()).intersection(act_frames.keys()))
    if not common_ids:
        return False, {"reason": "no_common_frames"}

    per_frame_scores: list[float] = []
    extra_total = 0
    missing_total = 0
    for fid in common_ids:
        exp_boxes = exp_frames[fid].get("boxes") or []
        act_boxes = act_frames[fid].get("boxes") or []
        if not exp_boxes and not act_boxes:
            per_frame_scores.append(1.0)
            continue
        used_act: set[int] = set()
        ious: list[float] = []
        for ebox in exp_boxes:
            best = 0.0
            best_idx = None
            for i, abox in enumerate(act_boxes):
                if i in used_act:
                    continue
                if int(abox.get("cls_id", -1)) != int(ebox.get("cls_id", -1)):
                    continue
                v = _iou(ebox, abox)
                if v > best:
                    best = v
                    best_idx = i
            if best_idx is not None:
                used_act.add(best_idx)
            ious.append(best)
        matched = len(used_act)
        missing = max(0, len(exp_boxes) - matched)
        extra = max(0, len(act_boxes) - matched)
        missing_total += missing
        extra_total += extra
        frame_score = (sum(ious) / len(exp_boxes)) if exp_boxes else 0.0
        per_frame_scores.append(frame_score)

    mean_iou = sum(per_frame_scores) / len(per_frame_scores)
    ok = mean_iou >= threshold
    return ok, {
        "mean_iou": mean_iou,
        "threshold": threshold,
        "extra_boxes": extra_total,
        "missing_boxes": missing_total,
        "frames_compared": len(common_ids),
    }


async def _load_latest_winners_snapshot() -> tuple[int, dict[str, Any]]:
    idx_url = get_settings().SCOREVISION_WINNERS_INDEX_URL
    keys = await fetch_json_from_url(idx_url)
    if not isinstance(keys, list) or not keys:
        raise RuntimeError("winners index is empty or invalid")
    latest_key = str(keys[-1])
    base = extract_base_url(idx_url)
    snapshot = await fetch_json_from_url(f"{base}/{latest_key}")
    if not isinstance(snapshot, dict):
        raise RuntimeError("latest winners snapshot invalid")
    return int(snapshot.get("block", 0) or 0), snapshot


def _manifest_public_element_ids(manifest: Any) -> set[str]:
    public_ids: set[str] = set()
    elems = getattr(manifest, "elements", None)
    if not elems:
        return public_ids

    if isinstance(elems, dict):
        for raw_eid, cfg in elems.items():
            track = cfg.get("track") if isinstance(cfg, dict) else getattr(cfg, "track", None)
            if track != "private":
                public_ids.add(str(raw_eid))
        return public_ids

    if isinstance(elems, (list, tuple)):
        for elem in elems:
            if isinstance(elem, dict):
                track = elem.get("track")
                eid = elem.get("element_id") or elem.get("id")
            else:
                track = getattr(elem, "track", None)
                eid = getattr(elem, "element_id", None) or getattr(elem, "id", None)
            if eid and track != "private":
                public_ids.add(str(eid))
    return public_ids


def _targets_from_winners(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    winners = snapshot.get("winners") or {}
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for element_id, entry in winners.items():
        if not isinstance(entry, dict):
            continue
        for key in ("top_3_official", "top_3_watchlist"):
            for row in (entry.get(key) or []):
                hk = str((row or {}).get("hotkey") or "").strip()
                eid = str(element_id)
                if not hk:
                    continue
                dedup_key = (eid, hk)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                commit_block = row.get("commit_block", row.get("block"))
                try:
                    commit_block = int(commit_block) if commit_block is not None else None
                except Exception:
                    commit_block = None
                targets.append(
                    {
                        "element_id": eid,
                        "hotkey": hk,
                        "model": row.get("model"),
                        "revision": row.get("revision"),
                        "commit_block": commit_block,
                    }
                )
    return sorted(targets, key=lambda t: (t["element_id"], t["hotkey"]))


def _pick_latest_miner_commit_for_element(
    commits: list[tuple[int, str]] | None,
    wanted_element_id: str,
) -> tuple[int | None, dict[str, Any] | None]:
    best_block = None
    best_obj = None
    for block, payload in (commits or []):
        try:
            block_i = int(block)
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("role") != "miner":
            continue
        committed_eid = obj.get("element_id")
        committed_eid = str(committed_eid).strip() if committed_eid is not None else None
        if committed_eid != wanted_element_id:
            continue
        if best_block is None or block_i > best_block:
            best_block = block_i
            best_obj = obj
    return best_block, best_obj


async def _fetch_commitment_context(
    netuid: int,
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, int]]:
    st = await get_subtensor()
    meta = await st.metagraph(netuid, mechid=get_settings().SCOREVISION_MECHID)
    commits = await st.get_all_revealed_commitments(netuid)
    hotkey_to_uid = {hk: uid for uid, hk in enumerate(meta.hotkeys)}
    return commits, hotkey_to_uid


async def _resolve_target_commit(
    target: dict[str, Any],
    commits_by_hotkey: dict[str, list[tuple[int, str]]],
    hotkey_to_uid: dict[str, int],
) -> dict[str, Any] | None:
    element_id = str(target["element_id"])
    hotkey = str(target["hotkey"])
    model = target.get("model")
    revision = target.get("revision")
    commit_block = target.get("commit_block")

    if model and revision and commit_block is not None:
        logger.info(
            "[compliance] target element=%s hotkey=%s resolved from winners snapshot block=%s",
            element_id,
            hotkey,
            commit_block,
        )
        return {
            "element_id": element_id,
            "hotkey": hotkey,
            "commit_block": int(commit_block),
            "model": str(model),
            "revision": str(revision),
            "uid": hotkey_to_uid.get(hotkey),
        }

    commits = commits_by_hotkey.get(hotkey)
    block, obj = _pick_latest_miner_commit_for_element(commits, element_id)
    if obj is None or block is None:
        logger.warning(
            "[compliance] target element=%s hotkey=%s no matching on-chain miner commitment",
            element_id,
            hotkey,
        )
        return None

    model = obj.get("model")
    revision = obj.get("revision")
    if not model or not revision:
        logger.warning(
            "[compliance] target element=%s hotkey=%s commitment missing model/revision",
            element_id,
            hotkey,
        )
        return None
    logger.info(
        "[compliance] target element=%s hotkey=%s resolved on-chain uid=%s block=%s model=%s revision=%s",
        element_id,
        hotkey,
        hotkey_to_uid.get(hotkey),
        block,
        model,
        revision,
    )
    return {
        "element_id": element_id,
        "hotkey": hotkey,
        "commit_block": int(block),
        "model": str(model),
        "revision": str(revision),
        "uid": hotkey_to_uid.get(hotkey),
    }


async def _sample_challenges_for_tuple(
    *,
    public_url: str,
    index_keys: list[str],
    element_id: str,
    hotkey: str,
    commit_block: int,
    k: int,
) -> list[dict[str, Any]]:
    keys = list(index_keys)
    random.shuffle(keys)
    safe_element_id = str(element_id).strip().replace("/", "_")
    prefix = f"manako/{safe_element_id}/{hotkey}/{max(0, int(commit_block)):09d}/evaluation/"
    candidates = [k for k in keys if isinstance(k, str) and k.startswith(prefix)]
    logger.info(
        "[compliance] sampling element=%s safe_element=%s hotkey=%s block=%s prefix_matches=%d",
        element_id,
        safe_element_id,
        hotkey,
        max(0, int(commit_block)),
        len(candidates),
    )
    random.shuffle(candidates)
    sampled = []
    seen_response_keys: set[str] = set()
    for key in candidates:
        lines = await fetch_shard_lines(public_url, key)
        for line in lines:
            payload = line.get("payload") or {}
            composite_score = payload.get("composite_score")
            try:
                score_val = float(composite_score)
            except Exception:
                score_val = 0.0
            if score_val <= 0.0:
                continue
            telemetry = payload.get("telemetry") or {}
            run_info = telemetry.get("run") or {}
            responses_key = run_info.get("responses_key")
            if not responses_key:
                continue
            responses_key = str(responses_key)
            if responses_key in seen_response_keys:
                continue
            seen_response_keys.add(responses_key)
            sampled.append(
                {
                    "challenge_id": str(
                        telemetry.get("challenge_id")
                        or telemetry.get("task_id")
                        or payload.get("challenge_id")
                        or payload.get("task_id")
                        or ""
                    ),
                    "responses_key": responses_key,
                }
            )
            if len(sampled) >= k:
                return sampled
    return sampled


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(0.95 * (len(ordered) - 1))))
    return float(ordered[idx])


async def run_public_compliance_once() -> dict[str, Any]:
    settings = get_settings()
    run_local_inference_from_hf = _get_security_runner()
    persistent_worker_class = _get_persistent_worker_class()
    cfg = checker_r2_config()
    if not is_configured(cfg, require_bucket=True):
        raise RuntimeError("Checker R2 not configured")

    st_for_manifest = await get_subtensor()
    current_block = int(await st_for_manifest.get_current_block())
    manifest = await load_manifest(path_manifest=None, settings=settings, block=current_block)
    public_element_ids = _manifest_public_element_ids(manifest)
    winners_block, snapshot = await _load_latest_winners_snapshot()
    raw_targets = _targets_from_winners(snapshot)
    targets = [t for t in raw_targets if str(t["element_id"]) in public_element_ids]
    skipped_non_public = len(raw_targets) - len(targets)
    logger.info(
        "[compliance] block=%s manifest_public_elements=%d winners_block=%s extracted_targets=%d public_targets=%d skipped_non_public=%d",
        current_block,
        len(public_element_ids),
        winners_block,
        len(raw_targets),
        len(targets),
        skipped_non_public,
    )
    if public_element_ids:
        logger.info(
            "[compliance] manifest public element ids sample: %s",
            ", ".join(sorted(public_element_ids)[:8]),
        )
    if targets:
        sample = ", ".join(
            f"{t['element_id']}:{t['hotkey'][:8]}...({('snapshot' if t.get('model') and t.get('revision') else 'chain')})"
            for t in targets[:5]
        )
        logger.info("[compliance] winners sample: %s", sample)
    commits_by_hotkey, hotkey_to_uid = await _fetch_commitment_context(settings.SCOREVISION_NETUID)
    index_keys = await fetch_index_keys(settings.SCOREVISION_PUBLIC_RESULTS_URL)
    logger.info("[compliance] loaded public index keys=%d", len(index_keys))

    run_results: list[dict[str, Any]] = []
    failed_tuples: list[dict[str, Any]] = []

    for target_row in targets:
        element_id = str(target_row["element_id"])
        hotkey = str(target_row["hotkey"])
        target = await _resolve_target_commit(target_row, commits_by_hotkey, hotkey_to_uid)
        if not target:
            run_results.append(
                {
                    "element_id": element_id,
                    "hotkey": hotkey,
                    "status": "FAIL_RUNTIME",
                    "reason": "target_commitment_missing",
                }
            )
            failed_tuples.append({"element_id": element_id, "hotkey": hotkey, "commit_block": None})
            continue

        sampled = await _sample_challenges_for_tuple(
            public_url=settings.SCOREVISION_PUBLIC_RESULTS_URL,
            index_keys=index_keys,
            element_id=element_id,
            hotkey=hotkey,
            commit_block=int(target["commit_block"]),
            k=max(
                settings.CHECKER_CHALLENGES_PER_TARGET,
                settings.CHECKER_CHALLENGES_PER_TARGET * 4,
            ),
        )
        if not sampled:
            logger.warning(
                "[compliance] no challenges found for element=%s hotkey=%s commit_block=%s",
                element_id,
                hotkey,
                target["commit_block"],
            )
            run_results.append(
                {
                    "element_id": element_id,
                    "hotkey": hotkey,
                    "commit_block": target["commit_block"],
                    "status": "FAIL_RUNTIME",
                    "reason": "no_challenges_found",
                }
            )
            failed_tuples.append(
                {"element_id": element_id, "hotkey": hotkey, "commit_block": target["commit_block"]}
            )
            continue

        desired_k = int(settings.CHECKER_CHALLENGES_PER_TARGET)
        latencies: list[float] = []
        all_ok = True
        details: list[dict[str, Any]] = []
        missing_blob_count = 0
        local_error_count = 0
        iou_fail_count = 0
        ok_count = 0
        worker = None
        if persistent_worker_class is not None:
            try:
                worker = persistent_worker_class(
                    model_repo=str(target["model"]),
                    revision=str(target["revision"]),
                    n_keypoints=32,
                    max_repo_bytes=settings.CHECKER_MAX_MODEL_BYTES,
                    memory_bytes=settings.CHECKER_RUNTIME_MEMORY_BYTES,
                    cpu_seconds=settings.CHECKER_RUNTIME_CPU_SECONDS,
                    wall_timeout_seconds=settings.CHECKER_RUNTIME_WALL_TIMEOUT_S,
                )
                worker.start()
                logger.info(
                    "[compliance] persistent worker started element=%s hotkey=%s model=%s revision=%s",
                    element_id,
                    hotkey,
                    target["model"],
                    target["revision"],
                )
            except Exception as e:
                logger.warning(
                    "[compliance] failed to start persistent worker element=%s hotkey=%s model=%s revision=%s err=%s; falling back",
                    element_id,
                    hotkey,
                    target["model"],
                    target["revision"],
                    e,
                )
                worker = None

        attempted = 0
        accepted = 0
        for record in sampled:
            if accepted >= desired_k:
                break
            attempted += 1
            challenge_id = record["challenge_id"]
            responses_key = record["responses_key"]
            exp_preds, _video_url, payload_frames = await fetch_responses_data(
                responses_key, settings.SCOREVISION_PUBLIC_RESULTS_URL
            )
            if not exp_preds or not payload_frames:
                missing_blob_count += 1
                logger.warning(
                    "[compliance] missing response blob element=%s hotkey=%s challenge_id=%s responses_key=%s has_preds=%s has_frames=%s",
                    element_id,
                    hotkey,
                    challenge_id,
                    responses_key,
                    bool(exp_preds),
                    bool(payload_frames),
                )
                details.append({"challenge_id": challenge_id, "ok": False, "reason": "missing_response_blob"})
                continue
            accepted += 1
            try:
                if worker is not None:
                    local = worker.infer(payload_frames=payload_frames, challenge_id=challenge_id)
                else:
                    local = run_local_inference_from_hf(
                        model_repo=str(target["model"]),
                        revision=str(target["revision"]),
                        payload_frames=payload_frames,
                        n_keypoints=32,
                        max_repo_bytes=settings.CHECKER_MAX_MODEL_BYTES,
                        memory_bytes=settings.CHECKER_RUNTIME_MEMORY_BYTES,
                        cpu_seconds=settings.CHECKER_RUNTIME_CPU_SECONDS,
                        wall_timeout_seconds=settings.CHECKER_RUNTIME_WALL_TIMEOUT_S,
                    )
            except Exception as e:
                all_ok = False
                local_error_count += 1
                logger.warning(
                    "[compliance] local run exception element=%s hotkey=%s challenge_id=%s model=%s revision=%s err=%s",
                    element_id,
                    hotkey,
                    challenge_id,
                    target["model"],
                    target["revision"],
                    e,
                )
                details.append({"challenge_id": challenge_id, "ok": False, "reason": f"local_exception:{e}"})
                continue
            if not local.success or not local.predictions:
                all_ok = False
                local_error_count += 1
                logger.warning(
                    "[compliance] local run failed element=%s hotkey=%s challenge_id=%s model=%s revision=%s err=%s",
                    element_id,
                    hotkey,
                    challenge_id,
                    target["model"],
                    target["revision"],
                    local.error,
                )
                details.append({"challenge_id": challenge_id, "ok": False, "reason": local.error})
                continue
            latencies.append(float(local.latency_ms))
            ok_iou, info = _compare_predictions_iou(
                expected=exp_preds,
                actual=local.predictions,
                threshold=settings.CHECKER_IOU_MATCH_THRESHOLD,
            )
            if not ok_iou:
                all_ok = False
                iou_fail_count += 1
                logger.warning(
                    "[compliance] iou mismatch element=%s hotkey=%s challenge_id=%s mean_iou=%.4f threshold=%.4f",
                    element_id,
                    hotkey,
                    challenge_id,
                    float(info.get("mean_iou", 0.0)),
                    float(info.get("threshold", settings.CHECKER_IOU_MATCH_THRESHOLD)),
                )
            else:
                ok_count += 1
            details.append(
                {
                    "challenge_id": challenge_id,
                    "ok": ok_iou,
                    "latency_ms": local.latency_ms,
                    "compare": info,
                }
            )

        if worker is not None:
            try:
                worker.close()
                logger.info("[compliance] persistent worker closed element=%s hotkey=%s", element_id, hotkey)
            except Exception as e:
                logger.warning("[compliance] persistent worker close error element=%s hotkey=%s err=%s", element_id, hotkey, e)

        if accepted == 0:
            all_ok = False
            logger.warning(
                "[compliance] no usable challenges after filtering/fetch element=%s hotkey=%s attempted=%d desired=%d",
                element_id,
                hotkey,
                attempted,
                desired_k,
            )

        p95_ms = _p95(latencies)
        latency_ok = p95_ms <= settings.CHECKER_LATENCY_P95_MS
        status = "PASS" if all_ok and latency_ok else ("FAIL_OUTPUT" if not all_ok else "FAIL_LATENCY")
        result_row = {
            "element_id": element_id,
            "hotkey": hotkey,
            "commit_block": target["commit_block"],
            "model": target["model"],
            "revision": target["revision"],
            "status": status,
            "p95_latency_ms": p95_ms,
            "latency_threshold_ms": settings.CHECKER_LATENCY_P95_MS,
            "details": details,
        }
        run_results.append(result_row)
        logger.info(
            "[compliance] target done element=%s hotkey=%s status=%s sampled=%d ok=%d local_errors=%d missing_blob=%d iou_fail=%d p95=%.2fms",
            element_id,
            hotkey,
            status,
            accepted,
            ok_count,
            local_error_count,
            missing_blob_count,
            iou_fail_count,
            float(p95_ms),
        )
        if status != "PASS":
            failed_tuples.append(
                {
                    "element_id": element_id,
                    "hotkey": hotkey,
                    "commit_block": target["commit_block"],
                    "status": status,
                }
            )

    payload = {
        "type": "public_compliance_run",
        "ts": time(),
        "winners_block": winners_block,
        "targets": len(targets),
        "results": run_results,
    }

    await ensure_index_exists(
        client_factory=_get_checker_client,
        bucket=cfg.bucket,
        index_key=_checker_runs_index_key(),
    )
    run_key = _checker_runs_key(winners_block)
    await _put_json(run_key, payload)
    await add_index_key_if_new(
        client_factory=_get_checker_client,
        bucket=cfg.bucket,
        key=run_key,
        index_key=_checker_runs_index_key(),
    )

    existing = await _get_json(_checker_fails_key())
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    if isinstance(existing, list):
        for row in existing:
            try:
                key = (str(row["hotkey"]), str(row["element_id"]), int(row["commit_block"]))
                merged[key] = row
            except Exception:
                continue
    now = time()
    for row in failed_tuples:
        cb = row.get("commit_block")
        if cb is None:
            continue
        key = (str(row["hotkey"]), str(row["element_id"]), int(cb))
        prev = merged.get(key)
        if prev is None:
            merged[key] = {
                "hotkey": key[0],
                "element_id": key[1],
                "commit_block": key[2],
                "first_seen": now,
                "last_seen": now,
                "latest_status": row.get("status"),
            }
        else:
            prev["last_seen"] = now
            prev["latest_status"] = row.get("status")

    await _put_json(_checker_fails_key(), list(merged.values()))
    return payload


async def compliance_loop() -> None:
    settings = get_settings()
    last_trigger_block = 0
    while True:
        try:
            st = await get_subtensor()
            block = int(await st.get_current_block())
            if (block - last_trigger_block) >= settings.CHECKER_INTERVAL_BLOCKS:
                out = await run_public_compliance_once()
                last_trigger_block = block
                logger.info(
                    "[compliance] run done winners_block=%s targets=%s",
                    out.get("winners_block"),
                    out.get("targets"),
                )
        except Exception as e:
            logger.warning("[compliance] loop error: %s", e)
        await asyncio.sleep(max(20, settings.CHECKER_POLL_INTERVAL_S))
