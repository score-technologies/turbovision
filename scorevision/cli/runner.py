from logging import getLogger
import os
import random
import asyncio
import signal
import gc
from pathlib import Path
from typing import Any

from scorevision.utils.settings import get_settings
from scorevision.utils.challenges import (
    get_challenge_from_scorevision,
    get_challenge_from_scorevision_with_source,
    prepare_challenge_payload,
    build_svchallenge_from_parts,
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
from scorevision.utils.chutes_helpers import warmup_chute, validate_chute_integrity
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

logger = getLogger(__name__)

# Global shutdown event for graceful shutdown
shutdown_event = asyncio.Event()


def _chute_id_for_miner(m: Miner) -> str | None:
    return getattr(m, "chute_id", None) or getattr(m, "slug", None)


async def _build_pgt_with_retries(
    chal_api: dict,
    *,
    required_n_frames: int,
    max_bbox_retries: int = 5,
    max_quality_retries: int = 5,
    video_cache: dict[str, Any] | None = None,
) -> tuple[SVChallenge, TVPredictInput, list]:
    """ """
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


async def runner(slug: str | None = None) -> None:
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
        os.getenv("SV_PGT_MAX_BBOX_RETRIES", os.getenv("SV_PGT_MAX_RETRIES", "5"))
    )
    MAX_PGT_QUALITY_RETRIES = int(os.getenv("SV_PGT_MAX_QUALITY_RETRIES", "5"))

    video_cache: dict[str, Any] = {}
    frame_store: FrameStore | None = None
    run_result = "success"
    try:
        miners = await get_miners_from_registry(NETUID)
        if not miners:
            logger.warning("No eligible miners found on-chain.")
            RUNNER_ACTIVE_MINERS.set(0)
            run_result = "no_miners"
            return

        challenge, payload, chal_api, frame_store = (
            await get_challenge_from_scorevision_with_source(video_cache=video_cache)
        )

        miner_list = list(miners.values())
        RUNNER_ACTIVE_MINERS.set(len(miner_list))

        try:
            pgt_build_start = loop.time()
            challenge, payload, pseudo_gt_annotations = await _build_pgt_with_retries(
                chal_api=chal_api,
                required_n_frames=REQUIRED_PGT_FRAMES,
                max_bbox_retries=MAX_PGT_BBOX_RETRIES,
                max_quality_retries=MAX_PGT_QUALITY_RETRIES,
                video_cache=video_cache,
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
            miner_output: TVPredictInput | None = None
            emission_started = False
            miner_total_start = loop.time()
            try:
                trustworthy = await validate_chute_integrity(chute_id=m.chute_id)
                if not trustworthy:
                    raise Exception(
                        "Chute integrity check failed. Skip it"
                    )  # NOTE: do we want to skip evaluating these miners or give them 0 scores?
                loop = asyncio.get_running_loop()
                start = loop.time()
                miner_output = await call_miner_model_on_chutes(
                    slug=m.slug,
                    chute_id=m.chute_id,
                    payload=payload,
                )
                latency_ms = (loop.time() - start) * 1000.0
                RUNNER_MINER_LATENCY_MS.labels(miner=miner_label).set(latency_ms)
                RUNNER_MINER_CALLS_TOTAL.labels(outcome="success").inc()
                logger.info(f"Miner {miner_output}")

                try:
                    evaluation = post_vlm_ranking(
                        payload=payload,
                        miner_run=miner_output,
                        challenge=challenge,
                        pseudo_gt_annotations=pseudo_gt_annotations,
                        frame_store=frame_store,
                    )
                except Exception:
                    RUNNER_EVALUATION_FAIL_TOTAL.labels(stage="ranking").inc()
                    raise
                logger.info(f"Evaluation: {evaluation}")
                if getattr(evaluation, "score", None) is not None:
                    RUNNER_EVALUATION_SCORE.labels(miner=miner_label).set(
                        getattr(evaluation, "score", 0.0)
                    )

                emission_started = True
                emit_start = loop.time()
                try:
                    await emit_shard(
                        slug=m.slug,
                        challenge=challenge,
                        miner_run=miner_output,
                        evaluation=evaluation,
                        miner_hotkey_ss58=m.hotkey,
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


async def runner_loop():
    """Runs `runner()` every 300 blocks, with robust triggering."""
    settings = get_settings()
    TEMPO = 300
    STALL_SECS_FALLBACK = 5400

    def signal_handler():
        logger.warning("Received shutdown signal, stopping runner...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    st = None
    last_trigger_block = None
    last_seen_block = None
    last_progress_time = asyncio.get_running_loop().time()

    logger.warning("[RunnerLoop] starting, TEMPO=%s blocks", TEMPO)

    while not shutdown_event.is_set():
        try:
            if st is None:
                logger.warning("[RunnerLoop] (re)connecting subtensor…")
                st = await get_subtensor()

            block = await st.get_current_block()
            RUNNER_BLOCK_HEIGHT.set(block)

            now = asyncio.get_running_loop().time()

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

            if should_trigger:
                logger.warning(
                    "[RunnerLoop] Triggering runner at block %s (last_trigger_block=%s)",
                    block,
                    last_trigger_block,
                )
                await runner()
                gc.collect()
                last_trigger_block = block
            else:
                try:
                    await asyncio.wait_for(st.wait_for_block(), timeout=30.0)
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
            st = None
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                pass

    logger.warning("Runner loop shutting down gracefully...")
