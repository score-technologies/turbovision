import asyncio
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from scorevision.utils.bittensor_helpers import get_subtensor, load_hotkey_keypair, reset_subtensor
from scorevision.utils.blacklist import BlacklistAPI, fetch_blacklisted_hotkeys
from scorevision.utils.docker_hub import fetch_image_digest
from scorevision.utils.manifest import Manifest
from scorevision.utils.r2 import add_index_key_if_new, central_r2_config, create_s3_client
from scorevision.utils.r2_public import fetch_index_keys, filter_keys_by_tail, fetch_shard_lines
from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.settings import get_settings
from scorevision.validator.central.private_track.challenges import (
    Challenge,
    get_challenge_with_ground_truth,
)
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner, get_registered_miners
from scorevision.validator.central.private_track.scoring import PRIVATE_SCORING_VERSION, score_predictions
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


async def _upload_shard(results: list[dict], block: int, hotkey_ss58: str) -> str | None:
    settings = get_settings()
    prefix = settings.PRIVATE_R2_RESULTS_PREFIX
    key = f"{prefix}/{block:09d}-{hotkey_ss58}.json"
    index_key = f"{prefix}/index.json"

    cfg = central_r2_config(settings)
    client_factory = lambda: create_s3_client(
        cfg, error_message="Central R2 not configured for private track"
    )

    try:
        async with client_factory() as client:
            await client.put_object(
                Bucket=cfg.bucket,
                Key=key,
                Body=dumps(results, separators=(",", ":")),
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
        headers = build_signed_headers(keypair, body) if keypair else {}
        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{api_url}/api/spotchecks/pending",
                content=body,
                headers=headers,
            )
            response.raise_for_status()
        logger.info("Pushed pending spotcheck for miner %s", spotcheck.miner_hotkey)
        return True
    except Exception as e:
        logger.error("Failed to push pending spotcheck: %s", e)
        return False


async def _challenge_miner(
    miner: RegisteredMiner,
    challenge: Challenge,
    keypair,
    timeout: float,
    block: int,
    image_digest: str,
) -> dict:
    try:
        attempt = await send_challenge(miner, challenge, keypair, timeout=timeout)
        response = attempt.response
        is_scored = response is not None and not attempt.timed_out

        if is_scored:
            score = score_predictions(response.predictions, challenge.ground_truth)
            pred_count = len(response.predictions)
        else:
            score = 0.0
            pred_count = 0

        if is_scored:
            logger.info(
                "Miner %s: score=%.3f response_time=%.3fs",
                miner.hotkey,
                score,
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
        ))
    except Exception as e:
        logger.error("Miner %s challenge processing failed: %s", miner.hotkey, e)
        return asdict(PrivateEvaluationResult(
            challenge_id=challenge.challenge_id,
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
        ))


async def _run_challenge_for_element(
    element_id: str,
    manifest: Manifest,
    block: int,
    keypair,
    subtensor,
) -> None:
    settings = get_settings()

    blacklist_api = None
    if settings.BLACKLIST_API_URL:
        blacklist_api = BlacklistAPI(settings.BLACKLIST_API_URL, keypair)

    metagraph = await subtensor.metagraph(settings.SCOREVISION_NETUID)
    blacklist = await fetch_blacklisted_hotkeys(blacklist_api)
    miners = await get_registered_miners(subtensor, metagraph, blacklist)

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

    digests = await asyncio.gather(*[
        fetch_image_digest(m.image_repo, m.image_tag) for m in miners
    ])
    results = list(await asyncio.gather(*[
        _challenge_miner(miner, challenge, keypair, settings.PRIVATE_MINER_TIMEOUT_S, block, digest)
        for miner, digest in zip(miners, digests)
    ]))

    await _upload_shard(results, block, keypair.ss58_address)


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
            entry["task"] = asyncio.create_task(
                _run_challenge_for_element(element_id, manifest, block, keypair, subtensor)
            )


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
