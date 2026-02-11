import asyncio
import os
from json import dumps
from logging import getLogger
from time import time
from bittensor import wallet
from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.r2 import (
    add_index_key_if_new,
    audit_r2_config,
    build_public_index_url_from_base,
    create_s3_client,
    ensure_index_exists,
    is_configured,
)
from scorevision.utils.settings import get_settings
from scorevision.utils.signing import _sign_batch
from scorevision.validator.models import ChallengeRecord, SpotcheckResult

logger = getLogger(__name__)


def _to_json_compatible(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _to_json_compatible(scalar())
        except Exception:
            pass
    return str(value)


def _audit_r2_enabled() -> bool:
    return is_configured(audit_r2_config(get_settings()), require_bucket=True)


def _audit_results_prefix() -> str:
    s = get_settings()
    ns = (s.AUDIT_R2_RESULTS_PREFIX or "audit_spotcheck").strip().strip("/")
    return f"scorevision/{ns}/"


def _safe_key_segment(value: str | None, fallback: str) -> str:
    s = (value or "").strip()
    if not s:
        return fallback
    return s.replace("/", "_")


def _get_audit_s3_client():
    cfg = audit_r2_config(get_settings())
    return create_s3_client(cfg, error_message="Audit R2 credentials not set")


def _build_audit_public_index_url() -> str | None:
    s = get_settings()
    return build_public_index_url_from_base(s.AUDIT_R2_BUCKET_PUBLIC_URL)


async def ensure_audit_index_exists() -> bool:
    if not _audit_r2_enabled():
        return False
    s = get_settings()
    return await ensure_index_exists(
        client_factory=_get_audit_s3_client,
        bucket=s.AUDIT_R2_BUCKET,
    )


async def _audit_index_add_if_new(key: str) -> None:
    s = get_settings()
    await add_index_key_if_new(
        client_factory=_get_audit_s3_client,
        bucket=s.AUDIT_R2_BUCKET,
        key=key,
    )


async def commit_audit_index_on_start() -> None:
    if os.getenv("AUDIT_COMMIT_VALIDATOR_ON_START", "1") in ("0", "false", "False"):
        logger.info("[audit-commit] Commit skipped by AUDIT_COMMIT_VALIDATOR_ON_START=0.")
        return

    index_url = _build_audit_public_index_url()
    if not index_url:
        logger.warning("[audit-commit] No AUDIT_R2_BUCKET_PUBLIC_URL configured; skipping commit.")
        return

    ok = await _commit_audit_index(index_url=index_url)
    if ok:
        logger.info("[audit-commit] On-chain commitment confirmed.")
    else:
        logger.warning("[audit-commit] On-chain commitment failed.")


async def _commit_audit_index(index_url: str) -> bool:
    s = get_settings()
    w = wallet(
        name=s.BITTENSOR_WALLET_COLD,
        hotkey=s.BITTENSOR_WALLET_HOT,
    )
    payload = {
        "role": "audit_validator",
        "hotkey": w.hotkey.ss58_address,
        "audit_index_url": index_url,
        "version": 1,
    }

    max_retries = s.AUDIT_COMMIT_MAX_RETRIES
    retry_delay_s = s.AUDIT_COMMIT_RETRY_DELAY_S

    for attempt in range(1, max_retries + 1):
        try:
            sub = await get_subtensor()
            await sub.set_reveal_commitment(
                wallet=w,
                netuid=s.SCOREVISION_NETUID,
                data=dumps(payload),
                blocks_until_reveal=1,
            )
            return True
        except Exception as e:
            logger.warning(
                "[audit-commit] attempt %d/%d failed: %s: %s",
                attempt,
                max_retries,
                type(e).__name__,
                e,
            )
            reset_subtensor()
            if attempt < max_retries:
                await asyncio.sleep(retry_delay_s)
    return False


def _build_spotcheck_key(record: ChallengeRecord, ts: int) -> str:
    block = record.block if record.block > 0 else ts
    element = _safe_key_segment(record.element_id, "unknown-element")
    miner = _safe_key_segment(record.miner_hotkey, "unknown-miner")
    challenge = _safe_key_segment(record.challenge_id, "unknown-challenge")
    return f"{_audit_results_prefix()}{element}/{miner}/spotcheck/{block:09d}-{challenge}-{ts}.json"


def _build_spotcheck_payload(
    record: ChallengeRecord,
    result: SpotcheckResult,
    *,
    mode: str,
    source: str,
    threshold: float | None,
    tail_blocks: int,
    mock_data_dir: str | None,
    timestamp: float | None = None,
) -> dict:
    return {
        "type": "audit_spotcheck",
        "timestamp": timestamp if timestamp is not None else time(),
        "mode": mode,
        "source": source,
        "tail_blocks": tail_blocks,
        "threshold": threshold,
        "challenge_id": record.challenge_id,
        "element_id": record.element_id,
        "window_id": record.window_id,
        "block": record.block,
        "miner_hotkey": record.miner_hotkey,
        "responses_key": record.responses_key,
        "scored_frame_numbers": record.scored_frame_numbers,
        "mock_data_dir": mock_data_dir,
        "result": {
            "passed": result.passed,
            "match_percentage": result.match_percentage,
            "central_score": result.central_score,
            "audit_score": result.audit_score,
            "details": result.details,
        },
    }


async def emit_spotcheck_result_shard(
    record: ChallengeRecord,
    result: SpotcheckResult,
    *,
    mode: str,
    source: str,
    threshold: float | None,
    tail_blocks: int,
    mock_data_dir: str | None = None,
) -> str | None:
    if not _audit_r2_enabled():
        logger.warning("[audit-r2] Audit R2 credentials not configured; skipping upload.")
        return None
    if not await ensure_audit_index_exists():
        logger.warning("[audit-r2] Could not ensure audit index; skipping upload.")
        return None

    s = get_settings()
    ts = int(time())
    key = _build_spotcheck_key(record, ts)
    payload = _build_spotcheck_payload(
        record=record,
        result=result,
        mode=mode,
        source=source,
        threshold=threshold,
        tail_blocks=tail_blocks,
        mock_data_dir=mock_data_dir,
    )
    payload = _to_json_compatible(payload)
    line = {"version": s.SCOREVISION_VERSION, "payload": payload}

    signed_line = dict(line)
    payload_str = dumps(payload, sort_keys=True, separators=(",", ":"))
    try:
        hk, _ = await _sign_batch([payload_str])
        if hk:
            signed_line["hotkey"] = hk
        else:
            logger.warning("[audit-r2] Signing returned empty; uploading unsigned shard.")
    except Exception as e:
        logger.warning("[audit-r2] Signing unavailable; uploading unsigned shard: %s", e)

    body = dumps([signed_line], separators=(",", ":"))
    async with _get_audit_s3_client() as c:
        await c.put_object(
            Bucket=s.AUDIT_R2_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    await _audit_index_add_if_new(key)

    logger.info("[audit-r2] Spotcheck shard uploaded: %s", key)
    return key
