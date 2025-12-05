from logging import getLogger
import os
import random
import asyncio
import signal
import gc
from pathlib import Path
from typing import Any, Optional

from scorevision.utils.settings import get_settings
from scorevision.utils.challenges import (
    get_challenge_from_scorevision,
    get_challenge_from_scorevision_with_source,
    prepare_challenge_payload,
    build_svchallenge_from_parts,
    ScoreVisionChallengeError,
)
from scorevision.utils.data_models import SVChallenge
from scorevision.chute_template.schemas import TVPredictInput
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.utils.async_clients import close_http_clients
from scorevision.vlm_pipeline.vlm_annotator import (
    generate_annotations_for_select_frames,
)
from scorevision.utils.miner_registry import get_miners_from_registry, Miner
from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import (
    filter_low_quality_pseudo_gt_annotations,
)
from scorevision.utils.manifest import Manifest, get_current_manifest
from scorevision.utils.chutes_helpers import warmup_chute
from scorevision.utils.prometheus import (
    RUNNER_BLOCK_HEIGHT,
    RUNNER_RUNS_TOTAL,
    RUNNER_WARMUP_TOTAL,
    RUNNER_PGT_RETRY_TOTAL,
    RUNNER_PGT_FRAMES,
    RUNNER_MINER_CALLS_TOTAL,
    RUNNER_MINER_LATENCY_MS,
    RUNNER_EVALUATION_SCORE,
    RUNNER_EVALUATION_FAIL_TOTAL,
    RUNNER_SHARDS_EMITTED_TOTAL,
    RUNNER_ACTIVE_MINERS,
    RUNNER_LAST_RUN_DURATION_SECONDS,
    RUNNER_LAST_PGT_DURATION_SECONDS,
    RUNNER_MINER_LAST_DURATION_SECONDS,
)
from scorevision.utils.video_processing import FrameStore
from scorevision.utils.windows import get_current_window_id, is_window_active
from scorevision.utils.commitments import get_active_element_ids_by_hotkey

logger = getLogger(__name__)

shutdown_event = asyncio.Event()


def _chute_id_for_miner(m: Miner) -> str | None:
    return getattr(m, "chute_id", None) or getattr(m, "slug", None)


async def _build_pgt_with_retries(
    chal_api: dict,
    manifest: Manifest,
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
                f"[PGT] Starting quality attempt {quality_attempt+1}/{max_quality_retries}"
            )

            for bbox_attempt in range(max_bbox_retries):
                try:
                    payload, frame_numbers, frames, flows, _frame_store = (
                        await prepare_challenge_payload(
                            challenge=chal_api,
                            video_cache=video_cache,
                        )
                    )

                    if len(frames) < required_n_frames:
                        logger.warning(
                            f"[PGT] Not enough frames ({len(frames)}/{required_n_frames}) "
                            f"bbox attempt {bbox_attempt+1}/{max_bbox_retries}"
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
                        await generate_annotations_for_select_frames(
                            video_name=challenge.challenge_id,
                            frames=challenge.frames,
                            flow_frames=challenge.dense_optical_flow_frames,
                            frame_numbers=challenge.frame_numbers,
                            manifest=manifest,
                        )
                    )
                    n_frames = len(pseudo_gt_annotations)
                    logger.info(
                        f"[PGT] {n_frames} pseudo-GT annotations generated "
                        f"(bbox attempt {bbox_attempt+1}/{max_bbox_retries})"
                    )

                    if not _enough_bboxes_per_frame(
                        pseudo_gt_annotations,
                        min_bboxes_per_frame=MIN_BBOXES_PER_FRAME,
                        min_frames_required=MIN_FRAMES_REQUIRED,
                    ):
                        logger.warning(
                            f"[PGT] Too few bboxes per frame. bbox retry "
                            f"{bbox_attempt+1}/{max_bbox_retries}"
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
                            f"(quality attempt {quality_attempt+1}/{max_quality_retries}, "
                            f"bbox attempt {bbox_attempt+1}/{max_bbox_retries})"
                        )
                        return challenge, payload, filtered

                    logger.warning(
                        f"[PGT] Not enough quality frames after filtering "
                        f"({len(filtered)}/{required_n_frames}), "
                        f"quality attempt {quality_attempt+1}/{max_quality_retries}, "
                        f"bbox attempt {bbox_attempt+1}/{max_bbox_retries}"
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="too_few_filtered").inc()

                except Exception as e:
                    last_err = e
                    logger.warning(
                        f"[PGT] Exception during bbox attempt {bbox_attempt+1}/{max_bbox_retries}: {e}"
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="exception").inc()
                    continue

            logger.warning(
                f"[PGT] Bbox phase failed after {max_bbox_retries} retries "
                f"→ new quality attempt ({quality_attempt+1}/{max_quality_retries})"
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


async def runner(
    slug: str | None = None,
    *,
    block_number: int | None = None,
    path_manifest: Path | None = None,
) -> None:
    settings = get_settings()
    loop = asyncio.get_running_loop()
    run_start = loop.time()
    last_pgt_duration = 0.0
    NETUID = settings.SCOREVISION_NETUID
    MAX_MINERS = int(os.getenv("SV_MAX_MINERS_PER_RUN", "60"))
    WARMUP_ENABLED = os.getenv("SV_WARMUP_BEFORE_RUN", "1") not in (
        "0",
        "false",
        "False",
    )
    WARMUP_CONC = int(os.getenv("SV_WARMUP_CONCURRENCY", "8"))
    WARMUP_TIMEOUT = int(os.getenv("SV_WARMUP_TIMEOUT_S", "60"))
    REQUIRED_PGT_FRAMES = int(getattr(settings, "SCOREVISION_VLM_SELECT_N_FRAMES", 3))
    MAX_PGT_RETRIES = int(os.getenv("SV_PGT_MAX_RETRIES", "3"))
    MAX_PGT_BBOX_RETRIES = int(
        os.getenv("SV_PGT_MAX_BBOX_RETRIES", os.getenv("SV_PGT_MAX_RETRIES", "3"))
    )
    MAX_PGT_QUALITY_RETRIES = int(os.getenv("SV_PGT_MAX_QUALITY_RETRIES", "4"))

    video_cache: dict[str, Any] = {}
    frame_store: FrameStore | None = None
    run_result = "success"

    use_v3 = os.getenv("SCOREVISION_USE_CHALLENGE_V3", "0") not in (
        "0",
        "false",
        "False",
    )
    manifest: Manifest | None = None
    manifest_hash: str | None = None
    expected_window_id: str | None = None

    try:
        if use_v3:
            try:
                if path_manifest is not None:
                    manifest = Manifest.load_yaml(path_manifest)
                else:
                    manifest = get_current_manifest(block_number=block_number)

                manifest_hash = manifest.hash
                expected_window_id = manifest.window_id

                blocks_to_expiry = None
                if manifest.expiry_block is not None and block_number is not None:
                    blocks_to_expiry = manifest.expiry_block - block_number

                logger.info(
                    "[Runner] Loaded Manifest: hash=%s window_id=%s expiry_block=%s blocks_to_expiry=%s",
                    manifest_hash,
                    expected_window_id,
                    getattr(manifest, "expiry_block", None),
                    blocks_to_expiry,
                )

                if block_number is not None and not is_window_active(
                    expected_window_id,
                    current_block=block_number,
                    expiry_block=manifest.expiry_block,
                ):
                    logger.warning(
                        "[Runner] Window %s is not active at block %s (expiry_block=%s). Skipping run.",
                        expected_window_id,
                        block_number,
                        manifest.expiry_block,
                    )
                    run_result = "window_inactive"
                    return

            except Exception as e:
                logger.error(
                    "[Runner] SCOREVISION_USE_CHALLENGE_V3=1 but failed to load Manifest: %s",
                    e,
                )
                run_result = "manifest_error"
                return

        miners = await get_miners_from_registry(NETUID)
        if not miners:
            logger.warning("No eligible miners found on-chain.")
            RUNNER_ACTIVE_MINERS.set(0)
            run_result = "no_miners"
            return

        try:
            if use_v3:
                challenge, payload, chal_api, frame_store = (
                    await get_challenge_from_scorevision_with_source(
                        video_cache=video_cache,
                        manifest_hash=manifest_hash,
                    )
                )

                chal_mh = chal_api.get("manifest_hash")
                chal_wid = chal_api.get("window_id")

                if manifest_hash and chal_mh and chal_mh != manifest_hash:
                    logger.warning(
                        "[Runner] Manifest hash mismatch between local Manifest (%s) and challenge (%s).",
                        manifest_hash,
                        chal_mh,
                    )

                if expected_window_id and chal_wid and chal_wid != expected_window_id:
                    logger.warning(
                        "[Runner] Window ID mismatch between local Manifest (%s) and challenge (%s).",
                        expected_window_id,
                        chal_wid,
                    )

                current_window_id = expected_window_id or chal_wid
            else:
                challenge, payload, chal_api, frame_store = (
                    await get_challenge_from_scorevision_with_source(
                        video_cache=video_cache
                    )
                )
                if block_number is not None:
                    current_window_id = get_current_window_id(block_number, tempo=300)
                else:
                    current_window_id = chal_api.get("window_id") or None

            logger.info("[Runner] Using window_id=%s for this run", current_window_id)

            element_id = None
            try:
                element_id = _extract_element_id_from_chal_api(chal_api)
            except Exception as e:
                logger.warning(
                    "[Runner] Failed to extract element_id from challenge: %s", e
                )

            if not element_id:
                logger.warning(
                    "[Runner] Challenge missing element_id; refusing to run without manifest-bound element. "
                    "Check challenge API / manifest wiring."
                )
                run_result = "no_element_id"
                return

            if not current_window_id:
                logger.warning(
                    "[Runner] No window_id associated with challenge; refusing to run."
                )
                run_result = "no_window_id"
                return

            logger.info(
                "[Runner] Challenge bound to element_id=%s window_id=%s",
                element_id,
                current_window_id,
            )
        except ScoreVisionChallengeError as ce:
            msg = str(ce)
            if "No active evaluation window" in msg:
                logger.warning(
                    "[Runner] No active evaluation window (404) from challenge API v3. Skipping run."
                )
                run_result = "no_window"
            elif "Rate limited by /api/challenge/v3" in msg:
                logger.warning(
                    "[Runner] Rate limited by challenge API v3 (409). Backing off this run."
                )
                run_result = "rate_limited"
            elif "Manifest expired or rejected" in msg:
                logger.warning(
                    "[Runner] Manifest expired or rejected (410). Operator action required."
                )
                run_result = "manifest_expired"
            else:
                logger.error("[Runner] Challenge API error: %s", msg)
                run_result = "challenge_error"
            return

        miner_list = list(miners.values())
        RUNNER_ACTIVE_MINERS.set(len(miner_list))

        try:
            active_commitments = await get_active_element_ids_by_hotkey(
                current_window_id
            )
        except Exception as e:
            logger.warning(
                "[Runner] Failed to load element commitments for window %s: %s",
                current_window_id,
                e,
            )
            active_commitments = {}

        if not active_commitments:
            logger.warning(
                "[Runner] No active miner element commitments found for window %s. "
                "Refusing to process this challenge. Miners must commit to elements via "
                "`sv miner` (or the appropriate commit helper) before participating.",
                current_window_id,
            )
            run_result = "no_element_commitments"
            return

        try:
            pgt_build_start = loop.time()
            challenge, payload, pseudo_gt_annotations = await _build_pgt_with_retries(
                chal_api=chal_api,
                required_n_frames=REQUIRED_PGT_FRAMES,
                max_bbox_retries=MAX_PGT_BBOX_RETRIES,
                max_quality_retries=MAX_PGT_QUALITY_RETRIES,
                video_cache=video_cache,
                manifest=manifest,
            )
            last_pgt_duration = loop.time() - pgt_build_start
            RUNNER_LAST_PGT_DURATION_SECONDS.set(last_pgt_duration)
        except Exception as e:
            logger.warning(
                f"PGT quality gating failed after retries, skipping challenge: {e}"
            )
            last_pgt_duration = loop.time() - pgt_build_start
            RUNNER_LAST_PGT_DURATION_SECONDS.set(last_pgt_duration)
            run_result = "pgt_failed"
            return

        for m in miner_list:
            miner_label = getattr(m, "slug", None) or str(getattr(m, "uid", "?"))

            hk = getattr(m, "hotkey", None)
            miner_commitments_for_hk = (
                active_commitments.get(hk or "", {}) if hk else {}
            )
            miner_proof = miner_commitments_for_hk.get(str(element_id))

            if not miner_proof:
                logger.info(
                    "[Runner] Skipping miner uid=%s slug=%s: no active commitment "
                    "for element_id=%s in window=%s",
                    getattr(m, "uid", "?"),
                    getattr(m, "slug", "?"),
                    element_id,
                    current_window_id,
                )
                RUNNER_MINER_CALLS_TOTAL.labels(outcome="skipped_no_commitment").inc()
                continue

            miner_output = None
            emission_started = False
            miner_total_start = loop.time()
            try:
                start = loop.time()
                miner_output = await call_miner_model_on_chutes(
                    slug=m.slug,
                    chute_id=m.chute_id,
                    payload=payload,
                )
                latency_ms = (loop.time() - start) * 1000.0
                RUNNER_MINER_LATENCY_MS.labels(miner=miner_label).set(latency_ms)
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
                emit_start = loop.time()

                commitment_meta = {
                    "window_id": miner_proof.window_id,
                    "element_ids": miner_proof.element_ids,
                    "model": miner_proof.model,
                    "revision": miner_proof.revision,
                    "chute_slug": miner_proof.chute_slug,
                    "chute_id": miner_proof.chute_id,
                    "service_cap": miner_proof.service_cap,
                    "commit_block": miner_proof.block,
                }

                try:
                    await emit_shard(
                        slug=m.slug,
                        challenge=challenge,
                        miner_run=miner_output,
                        evaluation=evaluation,
                        miner_hotkey_ss58=m.hotkey,
                        window_id=current_window_id,
                        element_id=str(element_id) if element_id is not None else None,
                        manifest_hash=manifest_hash,
                        salt_id=0,
                        pgt_recipe_hash=getattr(
                            settings, "SCOREVISION_PGT_RECIPE_HASH", None
                        ),
                        lane="public",
                        model=m.model,
                        revision=m.revision,
                        chute_id=m.chute_id,
                        commitment_meta=commitment_meta,
                    )
                except Exception:
                    dt_emit = (loop.time() - emit_start) * 1000.0
                    logger.exception(
                        "[emit] FAILED for %s in %.1fms", miner_label, dt_emit
                    )
                    raise
                else:
                    dt_emit = (loop.time() - emit_start) * 1000.0
                    logger.info("[emit] success for %s in %.1fms", miner_label, dt_emit)
                    RUNNER_SHARDS_EMITTED_TOTAL.labels(status="success").inc()
                finally:
                    emission_started = False
            except Exception as e:
                logger.warning(
                    "Miner uid=%s slug=%s failed: %s",
                    getattr(m, "uid", "?"),
                    getattr(m, "slug", "?"),
                    e,
                )
                if miner_output is None:
                    RUNNER_MINER_CALLS_TOTAL.labels(outcome="exception").inc()
                if emission_started:
                    RUNNER_SHARDS_EMITTED_TOTAL.labels(status="error").inc()
                continue
            finally:
                duration = loop.time() - miner_total_start
                RUNNER_MINER_LAST_DURATION_SECONDS.labels(miner=miner_label).set(
                    duration
                )
    except Exception as e:
        logger.error(e)
        run_result = "error"
    finally:
        loop_now = asyncio.get_running_loop()
        run_duration = loop_now.time() - run_start
        RUNNER_LAST_RUN_DURATION_SECONDS.set(run_duration)
        store_obj = video_cache.get("store") or frame_store
        if store_obj:
            try:
                store_obj.unlink()
            except Exception as err:
                logger.debug(
                    f"Failed to remove cached video {getattr(store_obj, 'video_path', '?')}: {err}"
                )
        elif video_cache.get("path"):
            cached_path = Path(video_cache["path"])
            try:
                cached_path.unlink(missing_ok=True)
            except Exception as err:
                logger.debug(f"Failed to remove cached video {cached_path}: {err}")
        video_cache.clear()
        RUNNER_RUNS_TOTAL.labels(result=run_result).inc()
        close_http_clients()
        gc.collect()


async def runner_loop(path_manifest: Path | None = None):
    settings = get_settings()
    TEMPO = 300
    STALL_SECS_FALLBACK = 5400
    GET_BLOCK_TIMEOUT = float(os.getenv("SUBTENSOR_GET_BLOCK_TIMEOUT_S", "15.0"))
    WAIT_BLOCK_TIMEOUT = float(os.getenv("SUBTENSOR_WAIT_BLOCK_TIMEOUT_S", "15.0"))
    RECONNECT_DELAY_S = float(os.getenv("SUBTENSOR_RECONNECT_DELAY_S", "5.0"))

    def signal_handler():
        logger.warning("Received shutdown signal, stopping runner...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    st = None
    last_trigger_block = None
    last_seen_block = None
    loop = asyncio.get_running_loop()
    last_progress_time = loop.time()
    last_trigger_time = loop.time()

    logger.warning("[RunnerLoop] starting, TEMPO=%s blocks", TEMPO)

    while not shutdown_event.is_set():
        try:
            if st is None:
                logger.warning("[RunnerLoop] (re)connecting subtensor…")
                try:
                    st = await get_subtensor()
                except Exception as e:
                    logger.warning(
                        "[RunnerLoop] subtensor connect failed: %s → retrying in %.1fs",
                        e,
                        RECONNECT_DELAY_S,
                    )
                    reset_subtensor()
                    st = None
                    await asyncio.sleep(RECONNECT_DELAY_S)
                    continue

            try:
                block = await asyncio.wait_for(
                    st.get_current_block(), timeout=GET_BLOCK_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[RunnerLoop] get_current_block() timed out after %.1fs → resetting subtensor",
                    GET_BLOCK_TIMEOUT,
                )
                reset_subtensor()
                st = None
                await asyncio.sleep(2.0)
                continue
            except (KeyError, ConnectionError, RuntimeError) as err:
                logger.warning(
                    "[RunnerLoop] get_current_block error (%s) → resetting subtensor",
                    err,
                )
                reset_subtensor()
                st = None
                await asyncio.sleep(2.0)
                continue

            RUNNER_BLOCK_HEIGHT.set(block)

            now = loop.time()

            if last_seen_block is None or block > last_seen_block:
                last_seen_block = block
                last_progress_time = now

            should_trigger = False
            if last_trigger_block is None:
                should_trigger = True
                logger.warning("[RunnerLoop] first trigger at block %s", block)
            else:
                if block - last_trigger_block >= TEMPO:
                    should_trigger = True

            if (now - last_progress_time) >= STALL_SECS_FALLBACK:
                logger.warning(
                    "[RunnerLoop] no block progress for %.0fs → fallback trigger",
                    now - last_progress_time,
                )
                should_trigger = True
                last_progress_time = now

            if (now - last_trigger_time) >= STALL_SECS_FALLBACK:
                logger.warning(
                    "[RunnerLoop] no run triggered for %.0fs (wall clock) → fallback trigger",
                    now - last_trigger_time,
                )
                should_trigger = True

            if should_trigger:
                logger.warning(
                    "[RunnerLoop] Triggering runner at block %s (last_trigger_block=%s)",
                    block,
                    last_trigger_block,
                )
                await runner(block_number=block, path_manifest=path_manifest)
                gc.collect()
                last_trigger_block = block
                last_trigger_time = loop.time()
            else:
                try:
                    await asyncio.wait_for(
                        st.wait_for_block(), timeout=WAIT_BLOCK_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning(
                        "[RunnerLoop] wait_for_block error (%s); resetting subtensor",
                        err,
                    )
                    reset_subtensor()
                    st = None
                    await asyncio.sleep(2.0)
                    continue

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(
                "[RunnerLoop] Error: %s; resetting subtensor and retrying…", e
            )
            reset_subtensor()
            st = None
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                pass

    logger.warning("Runner loop shutting down gracefully...")
