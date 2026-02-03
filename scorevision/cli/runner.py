import asyncio
import gc
import os
import signal
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Dict

from scorevision.chute_template.schemas import TVPredictInput
from scorevision.utils.async_clients import close_http_clients
from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.challenges import (
    ScoreVisionChallengeError,
    build_svchallenge_from_parts,
    get_challenge_from_scorevision_with_source,
    prepare_challenge_payload,
    complete_task_assignment,
    get_ground_truth_from_scorevision,
)
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.utils.data_models import SVChallenge
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.utils.manifest import Element, Manifest, get_current_manifest, load_manifest_from_public_index
from scorevision.utils.miner_registry import Miner, get_miners_from_registry
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.prometheus import (
    RUNNER_ACTIVE_MINERS,
    RUNNER_BLOCK_HEIGHT,
    RUNNER_EVALUATION_FAIL_TOTAL,
    RUNNER_LAST_PGT_DURATION_SECONDS,
    RUNNER_LAST_RUN_DURATION_SECONDS,
    RUNNER_MINER_CALLS_TOTAL,
    RUNNER_MINER_LAST_DURATION_SECONDS,
    RUNNER_MINER_LATENCY_MS,
    RUNNER_PGT_FRAMES,
    RUNNER_PGT_RETRY_TOTAL,
    RUNNER_RUNS_TOTAL,
    RUNNER_SHARDS_EMITTED_TOTAL,
)
from scorevision.utils.video_processing import FrameStore
from scorevision.utils.settings import get_settings
from scorevision.utils.windows import get_current_window_id, get_window_start_block
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import (
    filter_low_quality_pseudo_gt_annotations,
)
from scorevision.vlm_pipeline.vlm_annotator_sam3 import (
    generate_annotations_for_select_frames_sam3,
)

logger = getLogger(__name__)

shutdown_event = asyncio.Event()


def _chute_id_for_miner(m: Miner) -> str | None:
    return getattr(m, "chute_id", None) or getattr(m, "slug", None)


async def _build_pgt_with_retries(
    chal_api: dict,
    element: Element,
    *,
    required_n_frames: int,
    max_bbox_retries: int = 5,
    max_quality_retries: int = 5,
    video_cache: dict[str, Any] | None = None,
) -> tuple[SVChallenge, TVPredictInput, list]:
    created_local_cache = video_cache is None
    if video_cache is None:
        video_cache = {}

    MIN_BBOXES_PER_FRAME = int(os.getenv("SV_MIN_BBOXES_PER_FRAME", "6"))
    MIN_FRAMES_REQUIRED = int(
        os.getenv("SV_MIN_BBOX_FRAMES_REQUIRED", str(required_n_frames))
    )

    last_err = None

    try:
        for quality_attempt in range(max_quality_retries):
            logger.info(
                f"[PGT] Starting quality attempt {quality_attempt + 1}/{max_quality_retries}"
            )

            for bbox_attempt in range(max_bbox_retries):
                try:
                    (
                        payload,
                        frame_numbers,
                        frames,
                        flows,
                        _frame_store,
                    ) = await prepare_challenge_payload(
                        challenge=chal_api,
                        video_cache=video_cache,
                    )

                    if len(frames) < required_n_frames:
                        logger.warning(
                            f"[PGT] Not enough frames ({len(frames)}/{required_n_frames}) "
                            f"bbox attempt {bbox_attempt + 1}/{max_bbox_retries}"
                        )
                        RUNNER_PGT_RETRY_TOTAL.labels(
                            reason="insufficient_frames"
                        ).inc()
                        continue

                    challenge = build_svchallenge_from_parts(
                        chal_api=chal_api,
                        payload=payload,
                        frame_numbers=frame_numbers,
                        frames=frames,
                        flows=flows,
                    )

                    pseudo_gt_annotations = (
                        await generate_annotations_for_select_frames_sam3(
                            video_name=challenge.challenge_id,
                            frames=challenge.frames,
                            flow_frames=challenge.dense_optical_flow_frames,
                            frame_numbers=challenge.frame_numbers,
                            element=element,
                        )
                    )
                    n_frames = len(pseudo_gt_annotations)
                    logger.info(
                        f"[PGT] {n_frames} pseudo-GT annotations generated "
                        f"(bbox attempt {bbox_attempt + 1}/{max_bbox_retries})"
                    )

                    if not _enough_bboxes_per_frame(
                        pseudo_gt_annotations,
                        min_bboxes_per_frame=MIN_BBOXES_PER_FRAME,
                        min_frames_required=MIN_FRAMES_REQUIRED,
                    ):
                        logger.warning(
                            f"[PGT] Too few bboxes per frame. bbox retry "
                            f"{bbox_attempt + 1}/{max_bbox_retries}"
                        )
                        RUNNER_PGT_RETRY_TOTAL.labels(reason="too_few_bboxes").inc()
                        continue

                    filtered = filter_low_quality_pseudo_gt_annotations(
                        annotations=pseudo_gt_annotations
                    )
                    logger.info(f"[PGT] {len(filtered)} filtered annotations kept")

                    if _enough_bboxes_per_frame(
                        filtered,
                        min_bboxes_per_frame=MIN_BBOXES_PER_FRAME,
                        min_frames_required=required_n_frames,
                    ):
                        RUNNER_PGT_FRAMES.set(len(filtered))
                        logger.info(
                            f"[PGT] Success: enough filtered frames "
                            f"(quality attempt {quality_attempt + 1}/{max_quality_retries}, "
                            f"bbox attempt {bbox_attempt + 1}/{max_bbox_retries})"
                        )
                        return challenge, payload, filtered

                    logger.warning(
                        f"[PGT] Not enough quality frames after filtering "
                        f"({len(filtered)}/{required_n_frames}), "
                        f"quality attempt {quality_attempt + 1}/{max_quality_retries}, "
                        f"bbox attempt {bbox_attempt + 1}/{max_bbox_retries}"
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="too_few_filtered").inc()

                except Exception as e:
                    last_err = e
                    logger.warning(
                        f"[PGT] Exception during bbox attempt {bbox_attempt + 1}/{max_bbox_retries}: {e}"
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="exception").inc()
                    continue

            logger.warning(
                f"[PGT] Bbox phase failed after {max_bbox_retries} retries "
                f"→ new quality attempt ({quality_attempt + 1}/{max_quality_retries})"
            )
            RUNNER_PGT_RETRY_TOTAL.labels(reason="bbox_phase_failed").inc()

        raise RuntimeError(
            f"Failed to prepare high-quality PGT after {max_quality_retries} quality attempts "
            f"× {max_bbox_retries} bbox retries. Last error: {last_err}"
        )

    finally:
        if created_local_cache and video_cache:
            cached_path = video_cache.get("path")
            if cached_path:
                try:
                    from pathlib import Path as _Path

                    (
                        _Path(cached_path)
                        if not hasattr(cached_path, "unlink")
                        else cached_path
                    ).unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Failed to remove cached video {cached_path}: {e}")


def _enough_bboxes_per_frame(
    pseudo_gt_annotations: list,
    *,
    min_bboxes_per_frame: int,
    min_frames_required: int,
) -> bool:
    ok_frames = 0
    for pgt in pseudo_gt_annotations:
        n = len(getattr(pgt.annotation, "bboxes", []) or [])
        if n >= min_bboxes_per_frame:
            ok_frames += 1
    return ok_frames >= min_frames_required

def _to_pos_int(x: object) -> int | None:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int,)):
            return x if x > 0 else None
        if isinstance(x, float):
            xi = int(x)
            return xi if xi > 0 else None
        if isinstance(x, str):
            s = x.strip()
            if s.isdigit():
                xi = int(s)
                return xi if xi > 0 else None
    except Exception:
        pass
    return None

def _extract_element_id_from_chal_api(chal_api: dict) -> Optional[str]:
    if not isinstance(chal_api, dict):
        return None

    eid = chal_api.get("element_id")
    if isinstance(eid, str) and eid:
        return eid

    elem = chal_api.get("element") or {}
    if isinstance(elem, dict):
        eid = elem.get("element_id") or elem.get("id")
        if isinstance(eid, str) and eid:
            return eid

    meta = chal_api.get("meta") or {}
    if isinstance(meta, dict):
        eid = meta.get("element_id") or meta.get("element")
        if isinstance(eid, str) and eid:
            return eid

    return None

def _extract_element_tempos_from_manifest(
    manifest: Manifest,
    default_tempo_blocks: int,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    elems = getattr(manifest, "elements", None)

    if isinstance(elems, dict):
        for raw_eid, cfg in elems.items():
            eid = str(raw_eid)
            window_block = None
            if isinstance(cfg, dict):
                window_block = cfg.get("window_block") or cfg.get("tempo")
            else:
                window_block = getattr(cfg, "window_block", None) or getattr(cfg, "tempo", None)

            tempo = _to_pos_int(window_block) or default_tempo_blocks
            result[eid] = tempo
        return result

    if isinstance(elems, (list, tuple)):
        for elem in elems:
            if isinstance(elem, dict):
                eid = elem.get("element_id") or elem.get("id")
                window_block = elem.get("window_block") or elem.get("tempo")
            else:
                eid = getattr(elem, "element_id", None) or getattr(elem, "id", None)
                window_block = getattr(elem, "window_block", None) or getattr(elem, "tempo", None)

            if not eid:
                continue
            tempo = _to_pos_int(window_block) or default_tempo_blocks
            result[str(eid)] = tempo
        return result

    logger.warning("[RunnerLoop] Manifest 'elements' is neither dict nor list; no per-element scheduling.")
    return result


async def runner(
    slug: str | None = None,
    *,
    block_number: int | None = None,
    manifest: Manifest | None = None,
    element_id: str | None = None,
) -> None:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID
    required_pgt_frames = settings.SCOREVISION_VLM_SELECT_N_FRAMES
    max_pgt_bbox_retries = settings.RUNNER_PGT_MAX_BBOX_RETRIES
    max_pgt_quality_retries = settings.RUNNER_PGT_MAX_QUALITY_RETRIES
    default_element_tempo = settings.RUNNER_DEFAULT_ELEMENT_TEMPO

    event_loop = asyncio.get_running_loop()
    run_start = event_loop.time()
    video_cache: dict[str, Any] = {}
    frame_store: FrameStore | None = None
    run_result = "success"
    manifest_hash: str | None = None
    window_id: str | None = None

    logger.info("[Runner] START element_id=%s block=%s", element_id, block_number)

    try:
        chal_api: dict

        if manifest is None:
            logger.error("[Runner] No Manifest provided to runner()")
            run_result = "manifest_error"
            return
        if element_id is None:
            logger.error("[Runner] No element_id provided to runner()")
            run_result = "no_element_id"
            return

        manifest_hash = manifest.hash

        try:
            challenge, payload, chal_api, frame_store = (
                await get_challenge_from_scorevision_with_source(
                    video_cache=video_cache,
                    manifest_hash=manifest_hash,
                    element_id=element_id,
                )
            )
        except ScoreVisionChallengeError as ce:
            msg = str(ce)
            if "No active evaluation window" in msg:
                logger.warning(
                    "[Runner] (element=%s) No active evaluation window (404) from challenge API v3. Skipping run.",
                    element_id,
                )
                run_result = "no_window"
            elif "Rate limited by /api/challenge/v3" in msg:
                logger.warning(
                    "[Runner] (element=%s) Rate limited by challenge API v3 (409). Backing off this run.",
                    element_id,
                )
                run_result = "rate_limited"
            elif "Manifest expired or rejected" in msg:
                logger.warning(
                    "[Runner] (element=%s) Manifest expired or rejected (410). Operator action required.",
                    element_id,
                )
                run_result = "manifest_expired"
            else:
                logger.error("[Runner] Challenge API error: %s", msg)
                run_result = "challenge_error"
            return

        window_id = chal_api.get("window_id")

        eid_from_chal = _extract_element_id_from_chal_api(chal_api)
        if eid_from_chal and str(eid_from_chal) != str(element_id):
            logger.warning(
                "[Runner] (element=%s) element_id mismatch between requested (%s) and challenge (%s).",
                element_id,
                element_id,
                eid_from_chal,
            )

        if not element_id:
            logger.warning(
                "[Runner] Challenge missing element_id and no element_id forced. Refusing to run."
            )
            run_result = "no_element_id"
            return

        if not window_id:
            if block_number is None:
                logger.warning("[Runner] No window_id and no block_number; refusing to run.")
                run_result = "no_window_id"
                return
            window_id = get_current_window_id(block_number)
            logger.warning("[Runner] No window_id in challenge; derived window_id=%s from block=%s", window_id, block_number)

        logger.info("[Runner] Using window_id=%s for element_id=%s", window_id, element_id)

        tempo_by_element = _extract_element_tempos_from_manifest(manifest, default_element_tempo) if manifest else {}
        tempo_blocks = int(tempo_by_element.get(str(element_id), default_element_tempo))
        try:
            window_start_block = get_window_start_block(window_id, tempo=tempo_blocks)
        except Exception:
            window_start_block = block_number or 0

        logger.info("[Runner] window_start_block=%s (window_id=%s tempo=%s)", window_start_block, window_id, tempo_blocks)

        miners = await get_miners_from_registry(netuid, element_id=element_id)
        if not miners:
            logger.warning(
                "[Runner] No eligible miners found on-chain for element_id=%s.",
                element_id,
            )
            RUNNER_ACTIVE_MINERS.set(0)
            run_result = "no_miners"
            return

        miner_list = list[Miner](miners.values())
        RUNNER_ACTIVE_MINERS.set(len(miner_list))

        element = manifest.get_element(id=element_id)
        if element is None:
            raise ValueError(f"element id {element_id} not found in manifest")


        use_real_gt = bool(getattr(element, "ground_truth", False))

        if use_real_gt:
            challenge_id = int(chal_api.get("task_id"))

            try:
                await complete_task_assignment(challenge_id=challenge_id, element_id=element_id)

                gt = await get_ground_truth_from_scorevision(
                    challenge_id=challenge_id, element_id=element_id
                )
            except Exception as e:
                logger.warning(
                    f"[Runner] (element={element_id}) Ground-truth fetch failed, skipping challenge: {e}"
                )
                RUNNER_LAST_PGT_DURATION_SECONDS.set(0.0)
                run_result = "gt_failed"
                return

            pseudo_gt_annotations = gt
            RUNNER_LAST_PGT_DURATION_SECONDS.set(0.0)

        else:
            pgt_build_start = event_loop.time()
            try:
                challenge, payload, pseudo_gt_annotations = await _build_pgt_with_retries(
                    chal_api=chal_api,
                    required_n_frames=required_pgt_frames,
                    max_bbox_retries=max_pgt_bbox_retries,
                    max_quality_retries=max_pgt_quality_retries,
                    video_cache=video_cache,
                    element=element,
                )
            except Exception as e:
                logger.warning(f"[Runner] (element={element_id}) PGT quality gating failed: {e}")
                pgt_duration = event_loop.time() - pgt_build_start
                RUNNER_LAST_PGT_DURATION_SECONDS.set(pgt_duration)
                run_result = "pgt_failed"
                return

            pgt_duration = event_loop.time() - pgt_build_start
            RUNNER_LAST_PGT_DURATION_SECONDS.set(pgt_duration)

        for miner in miner_list:
            miner_label = miner.slug or str(getattr(miner, "uid", "?"))
            miner_output = None
            emission_started = False
            miner_start_time = event_loop.time()

            try:
                miner_output = await call_miner_model_on_chutes(
                    slug=miner.slug,
                    chute_id=miner.chute_id,
                    payload=payload,
                    expected_model=miner.model,
                    expected_revision=miner.revision,
                )
                RUNNER_MINER_LATENCY_MS.labels(miner=miner_label).set(miner_output.latency_ms)
                RUNNER_MINER_CALLS_TOTAL.labels(outcome="success").inc()

                try:
                    evaluation = post_vlm_ranking(
                        payload=payload,
                        miner_run=miner_output,
                        challenge=challenge,
                        pseudo_gt_annotations=pseudo_gt_annotations,
                        frame_store=frame_store,
                        manifest=manifest,
                        element_id=element_id,
                    )
                except Exception:
                    RUNNER_EVALUATION_FAIL_TOTAL.labels(stage="ranking").inc()
                    raise

                emission_started = True
                emit_start = event_loop.time()

                commitment_meta = {
                    "element_id": getattr(miner, "element_id", None),
                    "model": miner.model,
                    "revision": miner.revision,
                    "chute_slug": miner.slug,
                    "chute_id": miner.chute_id,
                    "commit_block": miner.block,
                    "window_id": window_id,
                }

                try:
                    await emit_shard(
                        slug=miner.slug,
                        challenge=challenge,
                        miner_run=miner_output,
                        evaluation=evaluation,
                        miner_hotkey_ss58=miner.hotkey,
                        window_id=window_id,
                        window_start_block=window_start_block,
                        trigger_block=block_number,
                        element_id=str(element_id) if element_id is not None else None,
                        manifest_hash=manifest_hash,
                        salt_id=0,
                        pgt_recipe_hash=getattr(settings, "SCOREVISION_PGT_RECIPE_HASH", None),
                        lane="public",
                        model=miner.model,
                        revision=miner.revision,
                        chute_id=miner.chute_id,
                        commitment_meta=commitment_meta,
                    )
                except Exception:
                    emit_duration_ms = (event_loop.time() - emit_start) * 1000.0
                    logger.exception("[emit] FAILED for %s in %.1fms", miner_label, emit_duration_ms)
                    raise
                else:
                    emit_duration_ms = (event_loop.time() - emit_start) * 1000.0
                    logger.info("[emit] success for %s in %.1fms", miner_label, emit_duration_ms)
                    RUNNER_SHARDS_EMITTED_TOTAL.labels(status="success").inc()
                finally:
                    emission_started = False

            except Exception as e:
                logger.warning("Miner uid=%s slug=%s failed: %s", getattr(miner, "uid", "?"), miner.slug, e)
                if miner_output is None:
                    RUNNER_MINER_CALLS_TOTAL.labels(outcome="exception").inc()
                if emission_started:
                    RUNNER_SHARDS_EMITTED_TOTAL.labels(status="error").inc()
                continue

            finally:
                miner_duration = event_loop.time() - miner_start_time
                RUNNER_MINER_LAST_DURATION_SECONDS.labels(miner=miner_label).set(miner_duration)
    except Exception as e:
        logger.error(e)
        run_result = "error"

    finally:
        run_duration = asyncio.get_running_loop().time() - run_start
        RUNNER_LAST_RUN_DURATION_SECONDS.set(run_duration)
        _cleanup_video_cache(video_cache, frame_store)
        RUNNER_RUNS_TOTAL.labels(result=run_result).inc()
        close_http_clients()
        gc.collect()


def _cleanup_video_cache(video_cache: dict[str, Any], frame_store: FrameStore | None) -> None:
    store_obj = video_cache.get("store") or frame_store
    if store_obj:
        try:
            store_obj.unlink()
        except Exception as err:
            logger.debug(f"Failed to remove cached video {getattr(store_obj, 'video_path', '?')}: {err}")
    elif video_cache.get("path"):
        cached_path = Path(video_cache["path"])
        try:
            cached_path.unlink(missing_ok=True)
        except Exception as err:
            logger.debug(f"Failed to remove cached video {cached_path}: {err}")
    video_cache.clear()


def _setup_shutdown_handler() -> None:
    def handler():
        logger.warning("Received shutdown signal, stopping runner...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: handler())


async def _load_manifest(path_manifest: Path | None, settings, block: int) -> Manifest:
    if path_manifest is not None:
        return Manifest.load_yaml(path_manifest)
    if getattr(settings, "URL_MANIFEST", None):
        cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
        return await load_manifest_from_public_index(settings.URL_MANIFEST, block_number=block, cache_dir=cache_dir)
    return get_current_manifest(block_number=block)


def _cancel_removed_element_tasks(element_state: Dict[str, Dict[str, Any]], element_tempos: Dict[str, int]) -> None:
    removed = set(element_state.keys()) - set(element_tempos.keys())
    for element_id in removed:
        entry = element_state.pop(element_id, None)
        if entry and entry.get("task") is not None:
            task = entry["task"]
            if not task.done():
                logger.info("[RunnerLoop] Cancelling runner task for removed element_id=%s", element_id)
                task.cancel()


def _update_element_state(element_state: Dict[str, Dict[str, Any]], element_tempos: Dict[str, int], block: int) -> None:
    for element_id, tempo in element_tempos.items():
        entry = element_state.get(element_id)
        window_id = get_current_window_id(block)
        anchor = get_window_start_block(window_id, tempo=tempo)

        if entry is None:
            element_state[element_id] = {"tempo": tempo, "anchor": anchor, "task": None}
            logger.info("[RunnerLoop] Registered new element_id=%s with tempo=%s blocks (anchor=%s)", element_id, tempo, anchor)
        else:
            entry["tempo"] = tempo
            entry["anchor"] = anchor


def _trigger_scheduled_runners(element_state: Dict[str, Dict[str, Any]], block: int, manifest: Manifest) -> None:
    for element_id, entry in element_state.items():
        tempo = max(1, int(entry["tempo"]))
        anchor = int(entry["anchor"])
        task = entry.get("task")
        delta = block - anchor
        should_trigger = (delta >= 0) and (delta % tempo == 0)

        if not should_trigger:
            continue

        if task is not None and not task.done():
            logger.info("[RunnerLoop] element_id=%s still running; skipping trigger at block=%s", element_id, block)
        else:
            logger.info("[RunnerLoop] Triggering runner for element_id=%s at block=%s (tempo=%s anchor=%s)", element_id, block, tempo, anchor)
            entry["task"] = asyncio.create_task(runner(block_number=block, manifest=manifest, element_id=element_id))


async def runner_loop(path_manifest: Path | None = None):
    settings = get_settings()
    get_block_timeout = settings.RUNNER_GET_BLOCK_TIMEOUT_S
    wait_block_timeout = settings.RUNNER_WAIT_BLOCK_TIMEOUT_S
    reconnect_delay = settings.RUNNER_RECONNECT_DELAY_S
    default_element_tempo = settings.RUNNER_DEFAULT_ELEMENT_TEMPO

    _setup_shutdown_handler()

    subtensor = None
    element_state: Dict[str, Dict[str, Any]] = {}
    manifest: Optional[Manifest] = None
    manifest_hash: Optional[str] = None

    logger.info("[RunnerLoop] starting (per-element scheduling)")

    while not shutdown_event.is_set():
        try:
            if subtensor is None:
                logger.info("[RunnerLoop] (re)connecting subtensor...")
                try:
                    subtensor = await get_subtensor()
                except Exception as e:
                    logger.warning("[RunnerLoop] subtensor connect failed: %s → retrying in %.1fs", e, reconnect_delay)
                    reset_subtensor()
                    subtensor = None
                    await asyncio.sleep(reconnect_delay)
                    continue

            try:
                block = await asyncio.wait_for(subtensor.get_current_block(), timeout=get_block_timeout)
            except asyncio.TimeoutError:
                logger.warning("[RunnerLoop] get_current_block() timed out after %.1fs → resetting", get_block_timeout)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue
            except (KeyError, ConnectionError, RuntimeError) as err:
                logger.warning("[RunnerLoop] get_current_block error (%s) → resetting subtensor", err)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue

            RUNNER_BLOCK_HEIGHT.set(block)

            try:
                new_manifest = await _load_manifest(path_manifest, settings, block)
            except Exception as e:
                logger.error("[RunnerLoop] Failed to load Manifest at block %s: %s", block, e)
                try:
                    await asyncio.wait_for(subtensor.wait_for_block(), timeout=wait_block_timeout)
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning("[RunnerLoop] wait_for_block error (%s); resetting subtensor", err)
                    reset_subtensor()
                    subtensor = None
                    await asyncio.sleep(2.0)
                continue

            new_hash = new_manifest.hash
            if manifest is None or new_hash != manifest_hash:
                logger.info("[RunnerLoop] Manifest hash changed (old=%s new=%s) → rebuilding element_state", manifest_hash, new_hash)
                manifest = new_manifest
                manifest_hash = new_hash

                element_tempos = _extract_element_tempos_from_manifest(manifest, default_element_tempo)
                _cancel_removed_element_tasks(element_state, element_tempos)
                _update_element_state(element_state, element_tempos, block)
            else:
                manifest = new_manifest

            if not element_state:
                logger.warning("[RunnerLoop] Manifest has no elements; nothing to schedule at block=%s", block)
            else:
                _trigger_scheduled_runners(element_state, block, manifest)

            try:
                await asyncio.wait_for(subtensor.wait_for_block(), timeout=wait_block_timeout)
            except asyncio.TimeoutError:
                continue
            except (KeyError, ConnectionError, RuntimeError) as err:
                logger.warning("[RunnerLoop] wait_for_block error (%s); resetting subtensor", err)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue

        except asyncio.CancelledError:
            break

        except Exception as e:
            logger.warning("[RunnerLoop] Error: %s; resetting subtensor and retrying...", e)
            reset_subtensor()
            subtensor = None
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                pass

    logger.info("Runner loop shutting down gracefully...")
