import asyncio
import random
from dataclasses import dataclass
from logging import getLogger
from typing import Any
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass
class ChallengeRecord:
    challenge_id: str
    element_id: str
    window_id: str
    block: int
    miner_hotkey: str
    central_score: float
    payload: dict[str, Any]


@dataclass
class SpotcheckResult:
    challenge_id: str
    element_id: str
    miner_hotkey: str
    central_score: float
    audit_score: float
    match_percentage: float
    passed: bool


def calculate_match_percentage(central_score: float, audit_score: float) -> float:
    if central_score == 0.0 and audit_score == 0.0:
        return 1.0
    if central_score == 0.0 or audit_score == 0.0:
        return 0.0
    diff = abs(central_score - audit_score)
    max_score = max(abs(central_score), abs(audit_score))
    return max(0.0, 1.0 - (diff / max_score))


def scores_match(central_score: float, audit_score: float, threshold: float) -> bool:
    return calculate_match_percentage(central_score, audit_score) >= threshold


async def fetch_random_challenge_record(
    tail_blocks: int,
    element_id: str | None = None,
) -> ChallengeRecord | None:
    logger.info("Selecting random challenge from R2 (tail=%d blocks)", tail_blocks)
    return None


async def fetch_challenge_payload(challenge_id: str) -> dict[str, Any] | None:
    logger.info("Fetching challenge payload for challenge=%s", challenge_id)
    return None


async def fetch_miner_results_for_challenge(
    challenge_id: str,
    element_id: str,
) -> list[dict[str, Any]]:
    logger.info("Fetching miner results for challenge=%s element=%s", challenge_id, element_id)
    return []


async def regenerate_ground_truth(
    challenge_payload: dict[str, Any],
    element_id: str,
) -> list[Any]:
    logger.info("Regenerating ground truth for element=%s", element_id)
    return []


async def rescore_miner_response(
    miner_response: dict[str, Any],
    ground_truth: list[Any],
    element_id: str,
) -> float:
    logger.info("Rescoring miner response for element=%s", element_id)
    return 0.0


async def run_spotcheck(
    challenge_record: ChallengeRecord,
    threshold: float | None = None,
) -> SpotcheckResult:
    settings = get_settings()
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    logger.info(
        "Running spotcheck for challenge=%s miner=%s",
        challenge_record.challenge_id,
        challenge_record.miner_hotkey,
    )

    challenge_payload = await fetch_challenge_payload(challenge_record.challenge_id)
    if challenge_payload is None:
        logger.warning("Could not fetch challenge payload for spotcheck")
        return SpotcheckResult(
            challenge_id=challenge_record.challenge_id,
            element_id=challenge_record.element_id,
            miner_hotkey=challenge_record.miner_hotkey,
            central_score=challenge_record.central_score,
            audit_score=0.0,
            match_percentage=0.0,
            passed=False,
        )

    ground_truth = await regenerate_ground_truth(
        challenge_payload,
        challenge_record.element_id,
    )

    miner_results = await fetch_miner_results_for_challenge(
        challenge_record.challenge_id,
        challenge_record.element_id,
    )

    miner_response = next(
        (r for r in miner_results if r.get("miner_hotkey") == challenge_record.miner_hotkey),
        None,
    )

    if miner_response is None:
        logger.warning("Could not find miner response for spotcheck")
        return SpotcheckResult(
            challenge_id=challenge_record.challenge_id,
            element_id=challenge_record.element_id,
            miner_hotkey=challenge_record.miner_hotkey,
            central_score=challenge_record.central_score,
            audit_score=0.0,
            match_percentage=0.0,
            passed=False,
        )

    audit_score = await rescore_miner_response(
        miner_response,
        ground_truth,
        challenge_record.element_id,
    )

    match_pct = calculate_match_percentage(challenge_record.central_score, audit_score)
    passed = match_pct >= threshold

    if not passed:
        logger.warning(
            "SPOTCHECK FAILED: challenge=%s miner=%s central=%.4f audit=%.4f match=%.2f%%",
            challenge_record.challenge_id,
            challenge_record.miner_hotkey,
            challenge_record.central_score,
            audit_score,
            match_pct * 100,
        )
    else:
        logger.info(
            "Spotcheck passed: challenge=%s miner=%s match=%.2f%%",
            challenge_record.challenge_id,
            challenge_record.miner_hotkey,
            match_pct * 100,
        )

    return SpotcheckResult(
        challenge_id=challenge_record.challenge_id,
        element_id=challenge_record.element_id,
        miner_hotkey=challenge_record.miner_hotkey,
        central_score=challenge_record.central_score,
        audit_score=audit_score,
        match_percentage=match_pct,
        passed=passed,
    )


def calculate_next_spotcheck_delay(
    min_interval_seconds: int,
    max_interval_seconds: int,
) -> float:
    return random.uniform(min_interval_seconds, max_interval_seconds)


async def spotcheck_loop(
    min_interval_seconds: int | None = None,
    max_interval_seconds: int | None = None,
    tail_blocks: int = 28800,
    threshold: float | None = None,
) -> None:
    settings = get_settings()

    if min_interval_seconds is None:
        min_interval_seconds = settings.AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if max_interval_seconds is None:
        max_interval_seconds = settings.AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    logger.info(
        "Starting spotcheck loop (interval=%d-%d seconds, threshold=%.0f%%)",
        min_interval_seconds,
        max_interval_seconds,
        threshold * 100,
    )

    while True:
        try:
            delay = calculate_next_spotcheck_delay(min_interval_seconds, max_interval_seconds)
            logger.info("[SpotcheckLoop] Next spotcheck in %.0f seconds", delay)
            await asyncio.sleep(delay)

            challenge_record = await fetch_random_challenge_record(tail_blocks)
            if challenge_record is None:
                logger.warning("[SpotcheckLoop] No challenge found for spotcheck")
                continue

            result = await run_spotcheck(challenge_record, threshold=threshold)
            logger.info(
                "[SpotcheckLoop] Spotcheck complete: passed=%s match=%.2f%%",
                result.passed,
                result.match_percentage * 100,
            )

        except asyncio.CancelledError:
            logger.info("[SpotcheckLoop] Cancelled, shutting down")
            break
        except Exception as e:
            logger.warning("[SpotcheckLoop] Error: %s", e)
            await asyncio.sleep(60)
