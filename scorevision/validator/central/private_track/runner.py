import asyncio
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from scorevision.miner.open_source.chute_template.schemas import TVPredictInput
from scorevision.utils.bittensor_helpers import get_subtensor, load_hotkey_keypair, reset_subtensor
from scorevision.utils.blacklist import BlacklistAPI, fetch_blacklisted_hotkeys
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.utils.data_models import SVChallenge, SVEvaluation, SVRunOutput
from scorevision.utils.manifest import Manifest
from scorevision.utils.r2 import R2Config, add_index_key_if_new, central_r2_config, create_s3_client
from scorevision.utils.r2_public import fetch_index_keys, filter_keys_by_tail, fetch_shard_lines
from scorevision.utils.signing import build_validator_query_params
from scorevision.utils.settings import get_settings
from scorevision.validator.central.private_track.challenges import (
    Challenge,
    get_challenge_with_ground_truth,
)
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner, get_registered_miners
from scorevision.validator.central.private_track.benchmark import (
    BenchmarkResult,
    compute_map_at_1s,
)
from scorevision.validator.central.private_track.scoring import (
    PRIVATE_SCORING_VERSION,
    score_predictions_with_breakdown,
)
from scorevision.validator.central.private_track.spotcheck import PendingSpotcheck
from scorevision.validator.central.scheduling import (
    cancel_removed_element_tasks,
    extract_element_tempos,
    load_manifest,
    setup_shutdown_handler,
    update_element_state,
)
from scorevision.validator.models import PrivateEvaluationResult

logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()

LOG_PREFIX = "[PTRunner] "


def _is_weight_eligible_result(result: dict) -> bool:
    return result.get("timed_out") is not True


def _private_responses_r2_config() -> R2Config:
    settings = get_settings()
    bucket = (settings.PRIVATE_RESPONSES_R2_BUCKET or settings.SCOREVISION_BUCKET or "").strip()
    account_id = (
        settings.PRIVATE_RESPONSES_R2_ACCOUNT_ID.get_secret_value()
        or settings.CENTRAL_R2_ACCOUNT_ID.get_secret_value()
    )
    access_key_id = (
        settings.PRIVATE_RESPONSES_R2_WRITE_ACCESS_KEY_ID.get_secret_value()
        or settings.CENTRAL_R2_WRITE_ACCESS_KEY_ID.get_secret_value()
    )
    secret_access_key = (
        settings.PRIVATE_RESPONSES_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
        or settings.CENTRAL_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
    )
    return R2Config(
        bucket=bucket,
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        concurrency=settings.CENTRAL_R2_CONCURRENCY,
    )


async def _upload_to_private_r2(key: str, index_key: str, payload: dict, label: str) -> str | None:
    cfg = _private_responses_r2_config()
    if not (cfg.bucket and cfg.account_id and cfg.access_key_id and cfg.secret_access_key):
        logger.warning("%sPrivate R2 not configured, skipping upload of %s", LOG_PREFIX, label)
        return None

    client_factory = lambda: create_s3_client(cfg, error_message="Private R2 is not configured")
    try:
        async with client_factory() as client:
            await client.put_object(
                Bucket=cfg.bucket,
                Key=key,
                Body=dumps(payload, separators=(",", ":")),
                ContentType="application/json",
            )
        await add_index_key_if_new(
            client_factory=client_factory,
            bucket=cfg.bucket,
            key=key,
            index_key=index_key,
        )
        return key
    except Exception as e:
        logger.error("%sFailed to upload %s: %s", LOG_PREFIX, label, e)
        return None


async def _upload_private_response_blob(
    *,
    element_id: str,
    miner: RegisteredMiner,
    challenge: Challenge,
    block: int,
    response_predictions: list[dict] | None,
) -> str | None:
    if not response_predictions:
        return None

    prefix = (get_settings().PRIVATE_RESPONSES_R2_PREFIX or "private_responses").strip().strip("/")
    safe_element = (element_id or "unknown-element").replace("/", "_")
    key = (
        f"{prefix}/{safe_element}/{miner.hotkey}/{max(0, int(miner.commit_block)):09d}/"
        f"responses/{block:09d}-{challenge.challenge_id}.json"
    )
    payload = {
        "track": "private",
        "element_id": element_id,
        "challenge_id": challenge.challenge_id,
        "video_url": challenge.video_url,
        "miner_hotkey": miner.hotkey,
        "miner_uid": miner.uid,
        "predictions": response_predictions,
    }
    return await _upload_to_private_r2(key, f"{prefix}/index.json", payload, f"response blob for miner {miner.hotkey}")


async def _upload_benchmark_result(
    *,
    element_id: str,
    miner: RegisteredMiner,
    challenge: Challenge,
    block: int,
    benchmark_result: BenchmarkResult,
) -> str | None:
    settings = get_settings()
    prefix = (settings.PRIVATE_BENCHMARK_R2_PREFIX or "private_benchmark").strip().strip("/")
    safe_element = (element_id or "unknown-element").replace("/", "_")
    key = (
        f"{prefix}/{safe_element}/{miner.hotkey}/{max(0, int(miner.commit_block)):09d}/"
        f"benchmark/{block:09d}-{challenge.challenge_id}.json"
    )
    payload = {
        "track": "private",
        "element_id": element_id,
        "challenge_id": challenge.challenge_id,
        "miner_hotkey": miner.hotkey,
        "miner_uid": miner.uid,
        "block": block,
        "benchmark_version": settings.BENCHMARK_VERSION,
        "map_at_1s": benchmark_result.map_at_1s,
        "per_action_ap": benchmark_result.per_action_ap,
    }
    return await _upload_to_private_r2(key, f"{prefix}/index.json", payload, f"benchmark for miner {miner.hotkey}")


_PUBLIC_SHARD_FIELDS = {
    "score", "challenge_id", "miner_hotkey", "miner_uid",
    "block", "timestamp", "image_repo", "image_tag",
    "image_digest", "scoring_version", "timed_out",
}


def _strip_for_public_shard(result: dict) -> dict:
    return {k: v for k, v in result.items() if k in _PUBLIC_SHARD_FIELDS}


async def _upload_shard(results: list[dict], block: int, hotkey_ss58: str) -> str | None:
    settings = get_settings()
    prefix = settings.PRIVATE_R2_RESULTS_PREFIX
    key = f"{prefix}/{block:09d}-{hotkey_ss58}.json"
    index_key = f"{prefix}/indexprivate.json"

    cfg = central_r2_config(settings)
    client_factory = lambda: create_s3_client(
        cfg, error_message="Central R2 not configured for private track"
    )

    try:
        async with client_factory() as client:
            await client.put_object(
                Bucket=cfg.bucket,
                Key=key,
                Body=dumps([_strip_for_public_shard(r) for r in results], separators=(",", ":")),
                ContentType="application/json",
            )
        await add_index_key_if_new(
            client_factory=client_factory,
            bucket=cfg.bucket,
            key=key,
            index_key=index_key,
        )
        logger.info("Uploaded shard: %s", key)
        return key
    except Exception as e:
        logger.error("Failed to upload shard: %s", e)
        return None


async def _fetch_recent_results(tail_blocks: int = 14400) -> list[dict]:
    settings = get_settings()
    public_index_url = settings.PRIVATE_R2_PUBLIC_INDEX_URL
    if not public_index_url:
        return []

    index_keys = await fetch_index_keys(public_index_url)
    if not index_keys:
        return []

    filtered, _, _ = filter_keys_by_tail(index_keys, tail_blocks)
    all_results: list[dict] = []
    for key in filtered:
        lines = await fetch_shard_lines(public_index_url, key)
        all_results.extend(lines)
    return all_results


async def _push_pending_spotcheck(spotcheck: PendingSpotcheck, keypair=None) -> bool:
    settings = get_settings()
    api_url = settings.PRIVATE_SPOTCHECK_API_URL
    if not api_url:
        logger.error("PRIVATE_SPOTCHECK_API_URL not configured")
        return False

    try:
        body = spotcheck.model_dump_json().encode()
        headers = {"Content-Type": "application/json"}
        params = build_validator_query_params(keypair) if keypair else None

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{api_url}/api/spotchecks/pending",
                params=params,
                content=body,
                headers=headers,
            )
            response.raise_for_status()
        logger.info("Pushed pending spotcheck for miner %s", spotcheck.miner_hotkey)
        return True
    except Exception as e:
        logger.error("Failed to push pending spotcheck: %s", e)
        return False


def _safe_compute_benchmark(
    predictions: list,
    ground_truth: list,
    miner_hotkey: str,
) -> BenchmarkResult | None:
    try:
        return compute_map_at_1s(predictions, ground_truth)
    except Exception as e:
        logger.error(
            "%sBenchmark computation failed for miner %s: %s",
            LOG_PREFIX,
            miner_hotkey,
            e,
        )
        return None


async def _challenge_miner(
    miner: RegisteredMiner,
    challenge: Challenge,
    keypair,
    timeout: float,
    block: int,
    element_id: str,
    pillar_weights: dict[str, float] | None,
    image_digest: str,
) -> tuple[dict, list[dict] | None, BenchmarkResult | None]:
    try:
        attempt = await send_challenge(miner, challenge, keypair, timeout=timeout)
        response = attempt.response
        is_scored = response is not None and not attempt.timed_out
        response_predictions = None
        benchmark_result = None

        if is_scored:
            score, score_breakdown = score_predictions_with_breakdown(
                response.predictions,
                challenge.ground_truth,
                pillar_weights=pillar_weights,
            )
            pred_count = len(response.predictions)
            response_predictions = [pred.model_dump() for pred in response.predictions]
            benchmark_result = _safe_compute_benchmark(
                response.predictions, challenge.ground_truth, miner.hotkey
            )
        else:
            score = 0.0
            pred_count = 0
            score_breakdown = {}

        if is_scored:
            benchmark_log = ""
            if benchmark_result is not None:
                benchmark_log = f" mAP@1s={benchmark_result.map_at_1s:.4f}"
            logger.info(
                "Miner %s: score=%.3f%s response_time=%.3fs",
                miner.hotkey,
                score,
                benchmark_log,
                attempt.elapsed_s,
            )
        else:
            logger.warning(
                "Miner %s excluded from private scoring response_time=%.3fs threshold=%.3fs",
                miner.hotkey,
                attempt.elapsed_s,
                timeout,
            )

        return asdict(PrivateEvaluationResult(
            challenge_id=challenge.challenge_id,
            element_id=element_id,
            miner_hotkey=miner.hotkey,
            miner_uid=miner.uid,
            score=score,
            prediction_count=pred_count,
            ground_truth_count=len(challenge.ground_truth),
            processing_time=attempt.elapsed_s,
            timestamp=datetime.now(timezone.utc).isoformat(),
            block=block,
            video_url=challenge.video_url,
            response_time_s=attempt.elapsed_s,
            timed_out=attempt.timed_out,
            image_repo=miner.image_repo,
            image_tag=miner.image_tag,
            image_digest=image_digest,
            scoring_version=PRIVATE_SCORING_VERSION,
            score_breakdown=score_breakdown,
        )), response_predictions, benchmark_result
    except Exception as e:
        logger.error("Miner %s challenge processing failed: %s", miner.hotkey, e)
        return asdict(PrivateEvaluationResult(
            challenge_id=challenge.challenge_id,
            element_id=element_id,
            miner_hotkey=miner.hotkey,
            miner_uid=miner.uid,
            score=0.0,
            prediction_count=0,
            ground_truth_count=len(challenge.ground_truth),
            processing_time=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            block=block,
            video_url=challenge.video_url,
            timed_out=True,
            image_repo=miner.image_repo,
            image_tag=miner.image_tag,
            image_digest=image_digest,
            scoring_version=PRIVATE_SCORING_VERSION,
            score_breakdown={},
        )), None, None


async def _emit_private_score_to_public_db(
    *,
    element_id: str,
    manifest_hash: str,
    challenge: Challenge,
    miner: RegisteredMiner,
    result: dict,
    private_responses_key: str | None,
    trigger_block: int,
) -> None:
    challenge_obj = SVChallenge(
        env="private-track",
        payload=TVPredictInput(url=challenge.video_url, meta={"track": "private"}),
        meta={
            "source": "private_track",
            "task_id": challenge.challenge_id,
        },
        prompt="private-track challenge",
        challenge_id=challenge.challenge_id,
        frame_numbers=[],
        frames=[],
        dense_optical_flow_frames=[],
        api_task_id=challenge.challenge_id,
    )

    latency_ms = float(result.get("response_time_s", 0.0)) * 1000.0
    timed_out = bool(result.get("timed_out", False))
    score = float(result.get("score", 0.0))
    score_breakdown = result.get("score_breakdown") or {"private_track_score": score}
    miner_run = SVRunOutput(
        success=not timed_out,
        latency_ms=latency_ms,
        predictions=None,
        error=None if not timed_out else "private-track timeout",
        model=f"{miner.image_repo}:{miner.image_tag}" if miner.image_repo and miner.image_tag else None,
        latency_p50_ms=latency_ms,
        latency_p95_ms=latency_ms,
        latency_p99_ms=latency_ms,
        latency_max_ms=latency_ms,
    )
    evaluation = SVEvaluation(
        acc_breakdown=score_breakdown,
        acc=score,
        latency_ms=latency_ms,
        score=score,
        details={
            "track": "private",
            "timed_out": timed_out,
            "prediction_count": result.get("prediction_count", 0),
            "ground_truth_count": result.get("ground_truth_count", 0),
        },
        latency_p95_ms=latency_ms,
        latency_pass=not timed_out,
        rtf=None,
        scored_frame_numbers=[gt.frame for gt in challenge.ground_truth],
    )
    commitment_meta = {
        "track": "private",
        "element_id": element_id,
        "commit_block": miner.commit_block,
        "image_repo": miner.image_repo,
        "image_tag": miner.image_tag,
        "image_digest": result.get("image_digest"),
    }

    await emit_shard(
        slug=f"private-{miner.uid}",
        challenge=challenge_obj,
        miner_run=miner_run,
        evaluation=evaluation,
        miner_hotkey_ss58=miner.hotkey,
        trigger_block=trigger_block,
        element_id=element_id,
        manifest_hash=manifest_hash,
        lane="private",
        model=miner.image_repo,
        revision=miner.image_tag,
        chute_id=None,
        commitment_meta=commitment_meta,
        commit_block=miner.commit_block,
        store_response_blob=False,
        responses_key_override=private_responses_key,
    )


def _log_runner_task_failure(task: asyncio.Task, element_id: str, block: int) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    logger.exception(
        "%sElement task failed (element_id=%s block=%s): %s",
        LOG_PREFIX,
        element_id,
        block,
        exc,
    )


async def _run_challenge_for_element(
    element_id: str,
    manifest: Manifest,
    block: int,
    keypair,
    subtensor,
) -> None:
    try:
        settings = get_settings()
        element = manifest.get_element(element_id)
        pillar_weights: dict[str, float] | None = None
        if element and element.metrics and element.metrics.pillars:
            pillar_weights = {
                str(k.value if hasattr(k, "value") else k): float(v)
                for k, v in element.metrics.pillars.items()
            }

        blacklist_api = None
        if settings.BLACKLIST_API_URL:
            blacklist_api = BlacklistAPI(settings.BLACKLIST_API_URL, keypair)

        metagraph = await subtensor.metagraph(settings.SCOREVISION_NETUID)
        blacklist = await fetch_blacklisted_hotkeys(blacklist_api)
        miners = await get_registered_miners(
            subtensor,
            metagraph,
            blacklist,
            element_id=element_id,
        )

        if not miners:
            logger.warning("%sNo registered private track miners for element=%s", LOG_PREFIX, element_id)
            return

        challenge = await get_challenge_with_ground_truth(
            manifest_hash=manifest.hash,
            element_id=element_id,
            keypair=keypair,
        )
        if not challenge:
            logger.warning("%sNo valid challenge for element=%s", LOG_PREFIX, element_id)
            return

        logger.info(
            "%sSending challenge %s to %d miners (element=%s)",
            LOG_PREFIX,
            challenge.challenge_id,
            len(miners),
            element_id,
        )

        outcomes = list(await asyncio.gather(*[
            _challenge_miner(
                miner,
                challenge,
                keypair,
                settings.PRIVATE_MINER_TIMEOUT_S,
                block,
                element_id,
                pillar_weights,
                miner.image_digest,
            )
            for miner in miners
        ]))

        results: list[dict] = []
        for miner, (result, response_predictions, benchmark_result) in zip(miners, outcomes):
            private_responses_key = await _upload_private_response_blob(
                element_id=element_id,
                miner=miner,
                challenge=challenge,
                block=block,
                response_predictions=response_predictions,
            )
            result["private_responses_key"] = private_responses_key
            results.append(result)

            if benchmark_result is not None:
                await _upload_benchmark_result(
                    element_id=element_id,
                    miner=miner,
                    challenge=challenge,
                    block=block,
                    benchmark_result=benchmark_result,
                )
            try:
                await _emit_private_score_to_public_db(
                    element_id=element_id,
                    manifest_hash=manifest.hash,
                    challenge=challenge,
                    miner=miner,
                    result=result,
                    private_responses_key=private_responses_key,
                    trigger_block=block,
                )
            except Exception as e:
                logger.error(
                    "%sFailed to emit private score to public db for miner=%s: %s",
                    LOG_PREFIX,
                    miner.hotkey,
                    e,
                )

        await _upload_shard(results, block, keypair.ss58_address)
    except Exception:
        logger.exception(
            "%sUnhandled error while running element_id=%s at block=%s",
            LOG_PREFIX,
            element_id,
            block,
        )
        raise


def _trigger_scheduled_runners(
    element_state: Dict[str, Dict[str, Any]],
    block: int,
    manifest: Manifest,
    keypair,
    subtensor,
) -> None:
    for element_id, entry in element_state.items():
        tempo = max(1, int(entry["tempo"]))
        anchor = int(entry["anchor"])
        task = entry.get("task")
        delta = block - anchor
        should_trigger = (delta >= 0) and (delta % tempo == 0)

        if not should_trigger:
            continue

        if task is not None and not task.done():
            logger.info("%selement_id=%s still running; skipping at block=%s", LOG_PREFIX, element_id, block)
        else:
            logger.info("%sTriggering challenge for element_id=%s at block=%s", LOG_PREFIX, element_id, block)
            task = asyncio.create_task(
                _run_challenge_for_element(element_id, manifest, block, keypair, subtensor)
            )
            task.add_done_callback(lambda t, e=element_id, b=block: _log_runner_task_failure(t, e, b))
            entry["task"] = task


async def challenge_loop(path_manifest: Path | None = None) -> None:
    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )
    default_element_tempo = settings.RUNNER_DEFAULT_ELEMENT_TEMPO
    get_block_timeout = settings.RUNNER_GET_BLOCK_TIMEOUT_S
    wait_block_timeout = settings.RUNNER_WAIT_BLOCK_TIMEOUT_S
    reconnect_delay = settings.RUNNER_RECONNECT_DELAY_S

    setup_shutdown_handler(shutdown_event)

    subtensor = None
    element_state: Dict[str, Dict[str, Any]] = {}
    manifest: Optional[Manifest] = None
    manifest_hash: Optional[str] = None

    logger.info("%sStarting (manifest-driven scheduling)", LOG_PREFIX)

    while not shutdown_event.is_set():
        try:
            if subtensor is None:
                logger.info("%s(re)connecting subtensor...", LOG_PREFIX)
                try:
                    subtensor = await get_subtensor()
                except Exception as e:
                    logger.warning("%ssubtensor connect failed: %s → retrying in %.1fs", LOG_PREFIX, e, reconnect_delay)
                    reset_subtensor()
                    subtensor = None
                    await asyncio.sleep(reconnect_delay)
                    continue

            try:
                block = await asyncio.wait_for(subtensor.get_current_block(), timeout=get_block_timeout)
            except asyncio.TimeoutError:
                logger.warning("%sget_current_block() timed out → resetting", LOG_PREFIX)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue
            except (KeyError, ConnectionError, RuntimeError) as err:
                logger.warning("%sget_current_block error (%s) → resetting", LOG_PREFIX, err)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue

            try:
                new_manifest = await load_manifest(path_manifest, settings, block)
            except Exception as e:
                logger.error("%sFailed to load manifest at block %s: %s", LOG_PREFIX, block, e)
                try:
                    await asyncio.wait_for(subtensor.wait_for_block(), timeout=wait_block_timeout)
                except asyncio.TimeoutError:
                    continue
                except (KeyError, ConnectionError, RuntimeError) as err:
                    logger.warning("%swait_for_block error (%s); resetting", LOG_PREFIX, err)
                    reset_subtensor()
                    subtensor = None
                    await asyncio.sleep(2.0)
                continue

            new_hash = new_manifest.hash
            if manifest is None or new_hash != manifest_hash:
                logger.info("%sManifest hash changed (old=%s new=%s) → rebuilding element_state", LOG_PREFIX, manifest_hash, new_hash)
                manifest = new_manifest
                manifest_hash = new_hash
                element_tempos = extract_element_tempos(manifest, default_element_tempo, track_filter="private")
                cancel_removed_element_tasks(element_state, element_tempos, log_prefix=LOG_PREFIX)
                update_element_state(element_state, element_tempos, block, log_prefix=LOG_PREFIX)
            else:
                manifest = new_manifest

            if not element_state:
                logger.warning("%sNo private-track elements in manifest at block=%s", LOG_PREFIX, block)
            else:
                _trigger_scheduled_runners(element_state, block, manifest, keypair, subtensor)

            try:
                await asyncio.wait_for(subtensor.wait_for_block(), timeout=wait_block_timeout)
            except asyncio.TimeoutError:
                continue
            except (KeyError, ConnectionError, RuntimeError) as err:
                logger.warning("%swait_for_block error (%s); resetting", LOG_PREFIX, err)
                reset_subtensor()
                subtensor = None
                await asyncio.sleep(2.0)
                continue

        except asyncio.CancelledError:
            break

        except Exception as e:
            logger.warning("%sError: %s; resetting subtensor and retrying...", LOG_PREFIX, e)
            reset_subtensor()
            subtensor = None
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                pass

    logger.info("%sShutting down gracefully...", LOG_PREFIX)


async def spotcheck_loop() -> None:
    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )
    subtensor = await get_subtensor()

    blacklist_api = None
    if settings.BLACKLIST_API_URL:
        blacklist_api = BlacklistAPI(settings.BLACKLIST_API_URL, keypair)

    while True:
        try:
            metagraph = await subtensor.metagraph(settings.SCOREVISION_NETUID)
            blacklist = await fetch_blacklisted_hotkeys(blacklist_api)
            miners = await get_registered_miners(subtensor, metagraph, blacklist)
            registered_hotkeys = {m.hotkey: m for m in miners}

            if not registered_hotkeys:
                logger.info("No registered miners for spot check")
                await asyncio.sleep(settings.PRIVATE_SPOTCHECK_INTERVAL_S)
                continue

            results = await _fetch_recent_results(tail_blocks=14400)
            if not results:
                logger.info("No recent results for spot check")
                await asyncio.sleep(settings.PRIVATE_SPOTCHECK_INTERVAL_S)
                continue

            eligible = [
                r
                for r in results
                if r.get("miner_hotkey") in registered_hotkeys and _is_weight_eligible_result(r)
            ]
            if not eligible:
                logger.info("No eligible miners with recent results for spot check")
                await asyncio.sleep(settings.PRIVATE_SPOTCHECK_INTERVAL_S)
                continue

            chosen = random.choice(eligible)
            miner = registered_hotkeys[chosen["miner_hotkey"]]
            coldkey = metagraph.coldkeys[miner.uid]

            image_repo = chosen.get("image_repo", miner.image_repo)
            image_tag = chosen.get("image_tag", miner.image_tag)
            image_digest = chosen.get("image_digest", "")
            scoring_version = chosen.get("scoring_version", 0)

            spotcheck = PendingSpotcheck(
                datetime_spotcheck=datetime.now(timezone.utc),
                miner_hotkey=miner.hotkey,
                miner_coldkey=coldkey,
                miner_username=image_repo.split("/")[0] if image_repo else "",
                miner_image_repo=image_repo,
                miner_image_tag=image_tag,
                miner_image_digest=image_digest,
                scoring_version=scoring_version,
                challenge_id=chosen["challenge_id"],
                challenge_url=chosen.get("video_url", ""),
                original_score=float(chosen["score"]),
            )

            await _push_pending_spotcheck(spotcheck, keypair)

        except Exception as e:
            logger.error("Spot check error: %s", e)

        await asyncio.sleep(settings.PRIVATE_SPOTCHECK_INTERVAL_S)


def run_challenge_process() -> None:
    asyncio.run(challenge_loop())


def run_spotcheck_process() -> None:
    asyncio.run(spotcheck_loop())
