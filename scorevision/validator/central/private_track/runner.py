import asyncio
import logging
import random
from dataclasses import asdict
from datetime import datetime, timezone
from json import dumps
import httpx
from scorevision.utils.bittensor_helpers import get_subtensor, load_hotkey_keypair
from scorevision.utils.blacklist import BlacklistAPI
from scorevision.utils.r2 import add_index_key_if_new, central_r2_config, create_s3_client
from scorevision.utils.r2_public import fetch_index_keys, filter_keys_by_tail, fetch_shard_lines
from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.settings import get_settings
from scorevision.validator.central.private_track.challenges import Challenge, select_challenge
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner, get_registered_miners
from scorevision.validator.central.private_track.scoring import score_predictions
from scorevision.validator.central.private_track.spotcheck import PendingSpotcheck
from scorevision.validator.models import PrivateEvaluationResult

logger = logging.getLogger(__name__)


async def fetch_video_segments(api_url: str, keypair=None) -> list[dict]:
    if not api_url:
        return []

    try:
        headers = build_signed_headers(keypair) if keypair else {}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{api_url}/api/segments", headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.error("Failed to fetch video segments: %s", e)
        return []

    return data.get("segments", [])


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


async def _get_blacklist(blacklist_api: BlacklistAPI | None) -> set[str]:
    if blacklist_api is None:
        return set()
    try:
        return await blacklist_api.get_blacklist()
    except Exception as e:
        logger.warning("Failed to fetch blacklist: %s", e)
        return set()


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
) -> dict:
    response = await send_challenge(miner, challenge, keypair, timeout=timeout)

    if response:
        score = score_predictions(response.predictions, challenge.ground_truth)
        pred_count = len(response.predictions)
        proc_time = response.processing_time
    else:
        score = 0.0
        pred_count = 0
        proc_time = 0.0

    logger.info("Miner %s: score=%.3f", miner.hotkey, score)

    return asdict(PrivateEvaluationResult(
        challenge_id=challenge.challenge_id,
        miner_hotkey=miner.hotkey,
        miner_uid=miner.uid,
        score=score,
        prediction_count=pred_count,
        ground_truth_count=len(challenge.ground_truth),
        processing_time=proc_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
        block=block,
        video_url=challenge.video_url,
    ))


async def challenge_loop() -> None:
    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )
    subtensor = await get_subtensor()

    blacklist_api = None
    if settings.PRIVATE_BLACKLIST_API_URL:
        blacklist_api = BlacklistAPI(settings.PRIVATE_BLACKLIST_API_URL, keypair)

    while True:
        try:
            metagraph = await subtensor.metagraph(settings.SCOREVISION_NETUID)
            blacklist = await _get_blacklist(blacklist_api)
            miners = await get_registered_miners(subtensor, metagraph, blacklist)

            if not miners:
                logger.warning("No registered private track miners")
                await asyncio.sleep(settings.PRIVATE_CHALLENGE_INTERVAL_S)
                continue

            segments = await fetch_video_segments(settings.PRIVATE_GT_API_URL, keypair)
            challenge = select_challenge(segments)
            if not challenge:
                logger.warning("No valid challenge available")
                await asyncio.sleep(settings.PRIVATE_CHALLENGE_INTERVAL_S)
                continue

            logger.info(
                "Sending challenge %s to %d miners",
                challenge.challenge_id,
                len(miners),
            )

            block = int(await subtensor.get_current_block())
            results = list(await asyncio.gather(*[
                _challenge_miner(miner, challenge, keypair, settings.PRIVATE_MINER_TIMEOUT_S, block)
                for miner in miners
            ]))

            await _upload_shard(results, block, keypair.ss58_address)

        except Exception as e:
            logger.error("Challenge loop error: %s", e)

        await asyncio.sleep(settings.PRIVATE_CHALLENGE_INTERVAL_S)


async def spotcheck_loop() -> None:
    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )
    subtensor = await get_subtensor()

    blacklist_api = None
    if settings.PRIVATE_BLACKLIST_API_URL:
        blacklist_api = BlacklistAPI(settings.PRIVATE_BLACKLIST_API_URL, keypair)

    while True:
        try:
            metagraph = await subtensor.metagraph(settings.SCOREVISION_NETUID)
            blacklist = await _get_blacklist(blacklist_api)
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

            eligible = [r for r in results if r.get("miner_hotkey") in registered_hotkeys]
            if not eligible:
                logger.info("No eligible miners with recent results for spot check")
                await asyncio.sleep(settings.PRIVATE_SPOTCHECK_INTERVAL_S)
                continue

            chosen = random.choice(eligible)
            miner = registered_hotkeys[chosen["miner_hotkey"]]
            coldkey = metagraph.coldkeys[miner.uid]

            spotcheck = PendingSpotcheck(
                datetime_spotcheck=datetime.now(timezone.utc),
                miner_hotkey=miner.hotkey,
                miner_coldkey=coldkey,
                miner_username=miner.image_repo.split("/")[0] if miner.image_repo else "",
                miner_image_repo=miner.image_repo,
                miner_image_tag=miner.image_tag,
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
