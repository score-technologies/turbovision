import asyncio
import random
from collections import defaultdict
from json import dumps
from logging import getLogger
from time import time
import httpx
from scorevision.utils.r2_public import fetch_index_keys, filter_keys_by_tail, fetch_shard_lines
from scorevision.utils.request_signing import build_signed_headers
from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction
from scorevision.utils.settings import get_settings
from scorevision.utils.signing import _sign_batch
from scorevision.validator.central.private_track.challenges import fetch_ground_truth
from scorevision.validator.central.private_track.scoring import (
    score_cricket_prediction_with_breakdown,
    score_predictions,
)
from scorevision.validator.models import SpotcheckResult

logger = getLogger(__name__)

LOG_PREFIX = "[PTSpotcheck] "


def calculate_match_percentage(central_score: float, audit_score: float) -> float:
    if central_score == 0.0 and audit_score == 0.0:
        return 1.0
    if central_score == 0.0 or audit_score == 0.0:
        return 0.0
    diff = abs(central_score - audit_score)
    max_score = max(abs(central_score), abs(audit_score))
    return max(0.0, 1.0 - (diff / max_score))


async def fetch_random_challenge(tail_blocks: int = 28800) -> tuple[str, list[dict]] | None:
    settings = get_settings()
    public_index_url = settings.PRIVATE_R2_PUBLIC_INDEX_URL
    if not public_index_url:
        logger.warning("%sPRIVATE_R2_PUBLIC_INDEX_URL not configured", LOG_PREFIX)
        return None

    index_keys = await fetch_index_keys(public_index_url)
    if not index_keys:
        logger.warning("%sNo index keys found", LOG_PREFIX)
        return None

    filtered, _, _ = filter_keys_by_tail(index_keys, tail_blocks)
    if not filtered:
        logger.warning("%sNo keys within tail window of %d blocks", LOG_PREFIX, tail_blocks)
        return None

    all_results: list[dict] = []
    random.shuffle(filtered)
    for key in filtered[:20]:
        lines = await fetch_shard_lines(public_index_url, key)
        all_results.extend(lines)

    if not all_results:
        return None

    by_challenge: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        cid = r.get("challenge_id")
        if cid and not r.get("timed_out"):
            by_challenge[cid].append(r)

    if not by_challenge:
        return None

    challenge_id = random.choice(list(by_challenge.keys()))
    return challenge_id, by_challenge[challenge_id]


PredictionPayload = list[dict] | dict | None
GroundTruth = list[FramePrediction] | CricketDeliveryPrediction


async def fetch_miner_responses(challenge_id: str, keypair) -> dict[str, PredictionPayload]:
    settings = get_settings()
    api_url = settings.PRIVATE_MINER_RESPONSES_API_URL
    if not api_url:
        raise RuntimeError("PRIVATE_MINER_RESPONSES_API_URL is not configured")

    headers = build_signed_headers(keypair)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            f"{api_url}/api/private-track/responses/{challenge_id}",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    result: dict[str, PredictionPayload] = {}
    for entry in data.get("responses", []):
        hotkey = entry.get("miner_hotkey")
        if hotkey:
            if "prediction" in entry:
                result[hotkey] = entry.get("prediction")
            else:
                result[hotkey] = entry.get("predictions", [])
    return result


def _extract_prediction_payload(payload: PredictionPayload) -> PredictionPayload:
    if isinstance(payload, dict):
        if "prediction" in payload:
            return payload.get("prediction")
        if "predictions" in payload:
            return payload.get("predictions")
    return payload


def _parse_cricket_prediction(payload: PredictionPayload) -> CricketDeliveryPrediction | None:
    payload = _extract_prediction_payload(payload)
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            return None
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    return CricketDeliveryPrediction(**payload)


def _parse_frame_predictions(payload: PredictionPayload) -> list[FramePrediction]:
    payload = _extract_prediction_payload(payload)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    predictions = [
        FramePrediction(frame=p["frame"], action=p["action"], confidence=p.get("confidence", 1.0))
        for p in payload
        if isinstance(p, dict) and "frame" in p and "action" in p
    ]
    return predictions


def rescore_miner(predictions_raw: PredictionPayload, ground_truth: GroundTruth) -> float:
    if isinstance(ground_truth, CricketDeliveryPrediction):
        prediction = _parse_cricket_prediction(predictions_raw)
        score, _ = score_cricket_prediction_with_breakdown(prediction, ground_truth)
        return score

    predictions = _parse_frame_predictions(predictions_raw)
    return score_predictions(predictions, ground_truth)


async def run_private_spotcheck(
    challenge_id: str,
    challenge_results: list[dict],
    keypair,
    threshold: float,
) -> list[SpotcheckResult]:
    miner_responses = await fetch_miner_responses(challenge_id, keypair)
    ground_truth_by_element: dict[str | None, GroundTruth] = {}

    results: list[SpotcheckResult] = []
    for entry in challenge_results:
        miner_hotkey = entry.get("miner_hotkey", "")
        element_id = entry.get("element_id") or ""
        central_score = float(entry.get("score", 0.0))

        predictions_raw = miner_responses.get(miner_hotkey)
        if predictions_raw is None:
            logger.warning("%sNo response data for miner %s", LOG_PREFIX, miner_hotkey)
            continue

        fetch_element_id = element_id or None
        if fetch_element_id not in ground_truth_by_element:
            ground_truth_by_element[fetch_element_id] = await fetch_ground_truth(
                challenge_id,
                keypair,
                element_id=fetch_element_id,
            )
        ground_truth = ground_truth_by_element[fetch_element_id]
        audit_score = rescore_miner(predictions_raw, ground_truth)
        match_pct = calculate_match_percentage(central_score, audit_score)
        passed = match_pct >= threshold

        logger.info(
            "%sMiner %s: central=%.4f audit=%.4f match=%.2f%% passed=%s",
            LOG_PREFIX, miner_hotkey, central_score, audit_score, match_pct * 100, passed,
        )

        results.append(SpotcheckResult(
            challenge_id=challenge_id,
            element_id=element_id,
            miner_hotkey=miner_hotkey,
            central_score=central_score,
            audit_score=audit_score,
            match_percentage=match_pct,
            passed=passed,
            details={
                "image_repo": entry.get("image_repo", ""),
                "image_tag": entry.get("image_tag", ""),
                "image_digest": entry.get("image_digest", ""),
                "scoring_version": entry.get("scoring_version", 0),
                "block": entry.get("block", 0),
            },
        ))

    return results


def _audit_r2_enabled() -> bool:
    from scorevision.utils.r2 import audit_r2_config, is_configured

    return is_configured(audit_r2_config(get_settings()), require_bucket=True)


def _audit_bucket() -> str:
    s = get_settings()
    return (s.AUDIT_R2_BUCKET or s.SCOREVISION_BUCKET or "").strip()


def _audit_prefix() -> str:
    return "manako/audit_private_spotcheck/"


def _get_audit_s3_client():
    from scorevision.utils.r2 import audit_r2_config, create_s3_client

    cfg = audit_r2_config(get_settings())
    return create_s3_client(cfg, error_message="Audit R2 credentials not set")


async def emit_private_spotcheck_results(
    results: list[SpotcheckResult],
    challenge_id: str,
    threshold: float,
) -> None:
    from scorevision.utils.r2 import add_index_key_if_new, ensure_index_exists

    if not _audit_r2_enabled():
        logger.warning("%sAudit R2 not configured; skipping upload", LOG_PREFIX)
        return

    index_key = f"{_audit_prefix()}index.json"
    await ensure_index_exists(
        client_factory=_get_audit_s3_client,
        bucket=_audit_bucket(),
        index_key=index_key,
    )

    ts = int(time())
    s = get_settings()

    for result in results:
        miner_segment = (result.miner_hotkey or "unknown").replace("/", "_")
        key = f"{_audit_prefix()}{challenge_id}/{miner_segment}-{ts}.json"

        payload = {
            "type": "audit_private_spotcheck",
            "timestamp": ts,
            "challenge_id": result.challenge_id,
            "miner_hotkey": result.miner_hotkey,
            "threshold": threshold,
            "result": {
                "passed": result.passed,
                "match_percentage": result.match_percentage,
                "central_score": result.central_score,
                "audit_score": result.audit_score,
                "details": result.details,
            },
        }

        line: dict = {"version": s.SCOREVISION_VERSION, "payload": payload}
        payload_str = dumps(payload, sort_keys=True, separators=(",", ":"))
        try:
            hk, sigs = await _sign_batch([payload_str])
            if hk and sigs:
                line["hotkey"] = hk
                line["signature"] = sigs[0]
        except Exception as e:
            logger.warning("%sSigning unavailable: %s", LOG_PREFIX, e)

        body = dumps([line], separators=(",", ":"))
        async with _get_audit_s3_client() as c:
            await c.put_object(
                Bucket=_audit_bucket(),
                Key=key,
                Body=body,
                ContentType="application/json",
            )

        await add_index_key_if_new(
            client_factory=_get_audit_s3_client,
            bucket=_audit_bucket(),
            key=key,
            index_key=index_key,
        )
        logger.info("%sUploaded spotcheck shard: %s", LOG_PREFIX, key)


async def private_spotcheck_loop(
    min_interval_seconds: int | None = None,
    max_interval_seconds: int | None = None,
    tail_blocks: int = 28800,
    threshold: float | None = None,
    commit_on_start: bool = True,
) -> None:
    from scorevision.utils.bittensor_helpers import load_hotkey_keypair

    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )

    if min_interval_seconds is None:
        min_interval_seconds = settings.PRIVATE_AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if max_interval_seconds is None:
        max_interval_seconds = settings.PRIVATE_AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.PRIVATE_SPOTCHECK_MATCH_THRESHOLD

    logger.info(
        "%sStarting loop (interval=%d-%ds, threshold=%.0f%%)",
        LOG_PREFIX, min_interval_seconds, max_interval_seconds, threshold * 100,
    )

    first_run = True

    while True:
        try:
            if first_run:
                first_run = False
                logger.info("%sRunning immediate first spotcheck", LOG_PREFIX)
            else:
                delay = random.uniform(min_interval_seconds, max_interval_seconds)
                logger.info("%sNext spotcheck in %.0f seconds", LOG_PREFIX, delay)
                await asyncio.sleep(delay)

            selected = await fetch_random_challenge(tail_blocks)
            if selected is None:
                logger.warning("%sNo challenge found for spotcheck", LOG_PREFIX)
                continue

            challenge_id, challenge_results = selected
            logger.info(
                "%sSpotchecking challenge %s (%d miner results)",
                LOG_PREFIX, challenge_id, len(challenge_results),
            )

            results = await run_private_spotcheck(challenge_id, challenge_results, keypair, threshold)

            passed_count = sum(1 for r in results if r.passed)
            logger.info(
                "%sSpotcheck complete: %d/%d miners passed",
                LOG_PREFIX, passed_count, len(results),
            )

            await emit_private_spotcheck_results(results, challenge_id, threshold)

        except asyncio.CancelledError:
            logger.info("%sCancelled, shutting down", LOG_PREFIX)
            break
        except Exception as e:
            logger.exception("%sError: %s", LOG_PREFIX, e)
            await asyncio.sleep(60)


async def run_single_private_spotcheck(
    tail_blocks: int = 28800,
    threshold: float | None = None,
) -> list[SpotcheckResult] | None:
    from scorevision.utils.bittensor_helpers import load_hotkey_keypair

    settings = get_settings()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD,
        settings.BITTENSOR_WALLET_HOT,
    )

    if threshold is None:
        threshold = settings.PRIVATE_SPOTCHECK_MATCH_THRESHOLD

    selected = await fetch_random_challenge(tail_blocks)
    if selected is None:
        logger.warning("%sNo challenge found", LOG_PREFIX)
        return None

    challenge_id, challenge_results = selected
    results = await run_private_spotcheck(challenge_id, challenge_results, keypair, threshold)
    await emit_private_spotcheck_results(results, challenge_id, threshold)
    return results
