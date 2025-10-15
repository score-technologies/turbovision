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
from scorevision.chute_template.schemas import SVPredictInput
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
    max_retries: int = 3,
    video_cache: dict[str, Any] | None = None,
) -> tuple[SVChallenge, SVPredictInput, list]:

    created_local_cache = video_cache is None
    if video_cache is None:
        video_cache = {}

    MAX_RETRIES_NOT_ENOUGH = int(os.getenv("SV_PGT_MAX_RETRIES_NOT_ENOUGH", str(max_retries)))
    MAX_RETRIES_QUALITY = int(os.getenv("SV_PGT_MAX_RETRIES_QUALITY", str(max_retries)))

    attempts_ne = 0
    attempts_q = 0

    last_err = None

    MIN_BBOXES_PER_FRAME = int(os.getenv("SV_MIN_BBOXES_PER_FRAME", "6"))
    MIN_FRAMES_REQUIRED = int(os.getenv("SV_MIN_BBOX_FRAMES_REQUIRED", str(required_n_frames)))

    try:
        while attempts_ne <= MAX_RETRIES_NOT_ENOUGH and attempts_q <= MAX_RETRIES_QUALITY:
            payload, frame_numbers, frames, flows, _frame_store = await prepare_challenge_payload(
                challenge=chal_api,
                video_cache=video_cache,
            )

            if len(frames) < required_n_frames:
                logger.warning(
                    f"Only {len(frames)} frames extracted (need >= {required_n_frames}). "
                    f"not_enough attempt {attempts_ne+1}/{MAX_RETRIES_NOT_ENOUGH} → resample."
                )
                RUNNER_PGT_RETRY_TOTAL.labels(reason="insufficient_frames").inc()
                attempts_ne += 1
                continue

            challenge = build_svchallenge_from_parts(
                chal_api=chal_api,
                payload=payload,
                frame_numbers=frame_numbers,
                frames=frames,
                flows=flows,
            )

            try:
                pseudo_gt_annotations = await generate_annotations_for_select_frames(
                    video_name=challenge.challenge_id,
                    frames=challenge.frames,
                    flow_frames=challenge.dense_optical_flow_frames,
                    frame_numbers=challenge.frame_numbers,
                )
                logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")

                if not _enough_bboxes_per_frame(
                    pseudo_gt_annotations,
                    min_bboxes_per_frame=MIN_BBOXES_PER_FRAME,
                    min_frames_required=MIN_FRAMES_REQUIRED,
                ):
                    logger.warning(
                        f"PGT has too few bboxes (need >= {MIN_BBOXES_PER_FRAME} on "
                        f"{MIN_FRAMES_REQUIRED} frames). not_enough attempt "
                        f"{attempts_ne+1}/{MAX_RETRIES_NOT_ENOUGH} → resample."
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="too_few_bboxes").inc()
                    attempts_ne += 1
                    continue

                filtered = filter_low_quality_pseudo_gt_annotations(
                    annotations=pseudo_gt_annotations
                )
                logger.info(f"{len(filtered)} Pseudo GT annotations had sufficient quality")

                if not _enough_bboxes_per_frame(
                    filtered,
                    min_bboxes_per_frame=MIN_BBOXES_PER_FRAME,
                    min_frames_required=required_n_frames,
                ):
                    logger.warning(
                        f"After quality filter, still too few bboxes on {required_n_frames} frames "
                        f"(need >= {MIN_BBOXES_PER_FRAME}). quality attempt "
                        f"{attempts_q+1}/{MAX_RETRIES_QUALITY} → resample."
                    )
                    RUNNER_PGT_RETRY_TOTAL.labels(reason="too_few_filtered").inc()
                    attempts_q += 1
                    continue

                if len(filtered) >= required_n_frames:
                    RUNNER_PGT_FRAMES.set(len(filtered))
                    return challenge, payload, filtered

                logger.warning(
                    f"Low-quality filter kept only {len(filtered)}/{required_n_frames} frames; "
                    f"quality attempt {attempts_q+1}/{MAX_RETRIES_QUALITY} → resample."
                )
                RUNNER_PGT_RETRY_TOTAL.labels(reason="insufficient_filtered_frames").inc()
                attempts_q += 1

            except Exception as e:
                last_err = e
                logger.warning(
                    f"PGT generation failed (exception) on quality attempt "
                    f"{attempts_q+1}/{MAX_RETRIES_QUALITY}: {e}. Retrying…"
                )
                RUNNER_PGT_RETRY_TOTAL.labels(reason="exception").inc()
                attempts_q += 1

        exhausted = []
        if attempts_ne > MAX_RETRIES_NOT_ENOUGH:
            exhausted.append(f"not_enough({attempts_ne}/{MAX_RETRIES_NOT_ENOUGH})")
        if attempts_q > MAX_RETRIES_QUALITY:
            exhausted.append(f"quality({attempts_q}/{MAX_RETRIES_QUALITY})")
        raise RuntimeError(
            "Failed to prepare PGT after retries: " + ", ".join(exhausted)
            + (f". Last error: {last_err}" if last_err else "")
        )
        
    finally:
        if created_local_cache and video_cache:
            cached_path = video_cache.get("path")
            if cached_path:
                try:
                    from pathlib import Path as _Path
                    (_Path(cached_path) if not hasattr(cached_path, "unlink") else cached_path).unlink(missing_ok=True)
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
            challenge, payload, pseudo_gt_annotations = await _build_pgt_with_retries(
                chal_api=chal_api,
                required_n_frames=REQUIRED_PGT_FRAMES,
                max_retries=MAX_PGT_RETRIES,
                video_cache=video_cache,
            )
        except Exception as e:
            logger.warning(
                f"PGT quality gating failed after retries, skipping challenge: {e}"
            )
            run_result = "pgt_failed"
            return

        for m in miner_list:
            miner_label = getattr(m, "slug", None) or str(getattr(m, "uid", "?"))
            miner_output: SVPredictInput | None = None
            emission_started = False
            try:
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
    except Exception as e:
        logger.error(e)
        run_result = "error"
    finally:
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
    """Runs `runner()` every N blocks (default: 300)."""
    settings = get_settings()
    TEMPO = 300

    # Set up signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal, stopping runner...")
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: signal_handler())

    st = None
    last_block = -1

    while not shutdown_event.is_set():
        try:
            if st is None:
                st = await get_subtensor()

            block = await st.get_current_block()
            RUNNER_BLOCK_HEIGHT.set(block)

            if block <= last_block or block % TEMPO != 0:
                try:
                    await asyncio.wait_for(st.wait_for_block(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning("wait_for_block error (%s); resetting subtensor", err)
                    reset_subtensor()
                    st = None
                    await asyncio.sleep(2.0)
                    continue
                continue

            logger.info(f"[RunnerLoop] Triggering runner at block {block}")
            await runner()
            gc.collect()

            last_block = block

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[RunnerLoop] Error: {e}; retrying…")
            st = None
            # Check shutdown event during sleep
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
                break
            except asyncio.TimeoutError:
                continue
    
    logger.info("Runner loop shutting down gracefully...")
