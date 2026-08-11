"""Publish a daily, randomly sampled audit of private-track winners.

The score shards and the resulting audit are public. Miner responses are read
from a separate private R2 bucket and are only disclosed for the selected
sample.
"""

import asyncio
import random
import signal
from json import dumps, loads
from logging import getLogger
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    load_hotkey_keypair,
    reset_subtensor,
)
from scorevision.utils.manifest import load_manifest_from_public_index
from scorevision.utils.r2 import (
    R2Config,
    add_index_key_if_new,
    central_r2_config,
    create_s3_client,
    ensure_index_exists,
)
from scorevision.utils.r2_public import (
    extract_block_from_key,
    extract_element_miner_commit_from_key,
    extract_base_url,
    fetch_json_from_url,
    fetch_shard_lines,
)
from scorevision.utils.settings import get_settings
from scorevision.utils.signing import build_validator_query_params

logger = getLogger(__name__)
LOG_PREFIX = "[PrivateAuditExport] "
RECENT_CHALLENGES_EXCLUDED = 10

shutdown_event = asyncio.Event()


def _prefix() -> str:
    return get_settings().PRIVATE_AUDIT_EXPORT_PREFIX.strip().strip("/") or "manako/audit"


def _index_key() -> str:
    return f"{_prefix()}/index.json"


def _run_key(trigger_block: int) -> str:
    return f"{_prefix()}/{max(0, int(trigger_block)):09d}.json"


def _block_from_key(key: str) -> int | None:
    """Extract a block from both `<block>.json` and `<block>-<id>.json`."""
    raw_path = urlparse(key).path if "://" in key else key
    stem = Path(raw_path).stem
    try:
        return int(stem.split("-", 1)[0])
    except (TypeError, ValueError):
        return None


def _private_responses_config() -> R2Config:
    settings = get_settings()
    return R2Config(
        bucket=settings.PRIVATE_RESPONSES_R2_BUCKET.strip(),
        account_id=settings.PRIVATE_RESPONSES_R2_ACCOUNT_ID.get_secret_value(),
        access_key_id=settings.PRIVATE_RESPONSES_R2_READ_ACCESS_KEY_ID.get_secret_value(),
        secret_access_key=settings.PRIVATE_RESPONSES_R2_READ_SECRET_ACCESS_KEY.get_secret_value(),
        concurrency=settings.CENTRAL_R2_CONCURRENCY,
    )


def _client_factory(config: R2Config, message: str):
    return lambda: create_s3_client(config, error_message=message)


async def _load_json_object(config: R2Config, key: str, *, label: str) -> Any:
    if not config.bucket:
        raise RuntimeError(f"{label} bucket is not configured")
    async with create_s3_client(
        config, error_message=f"{label} credentials are not configured"
    ) as client:
        response = await client.get_object(Bucket=config.bucket, Key=key)
        return loads((await response["Body"].read()).decode("utf-8"))


async def _fetch_index_keys(index_url: str) -> list[str]:
    data = await fetch_json_from_url(index_url)
    if isinstance(data, list):
        return [key for key in data if isinstance(key, str)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [
            entry["path"]
            for entry in data["entries"]
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        ]
    raise RuntimeError(f"Invalid or empty index: {index_url}")


async def load_latest_winners_snapshot() -> tuple[int, dict[str, Any]]:
    index_url = get_settings().SCOREVISION_WINNERS_INDEX_URL
    keys = await _fetch_index_keys(index_url)
    if not keys:
        raise RuntimeError("Winners index is empty")
    latest_key = max(keys, key=lambda key: _block_from_key(key) or -1)
    if latest_key.startswith(("http://", "https://")):
        snapshot_url = latest_key
    else:
        snapshot_url = f"{extract_base_url(index_url)}/{latest_key.lstrip('/')}"
    snapshot = await fetch_json_from_url(snapshot_url)
    if not isinstance(snapshot, dict):
        raise RuntimeError(f"Invalid winners snapshot: {latest_key}")
    return int(snapshot.get("block") or _block_from_key(latest_key) or 0), snapshot


async def _private_manifest_elements(block: int) -> dict[str, str]:
    settings = get_settings()
    index_url = settings.PRIVATE_AUDIT_MANIFEST_INDEX_URL
    if not index_url:
        raise RuntimeError("PRIVATE_AUDIT_MANIFEST_INDEX_URL is not configured")
    manifest = await load_manifest_from_public_index(
        index_url,
        block_number=block,
        cache_dir=settings.SCOREVISION_CACHE_DIR,
    )
    private: dict[str, str] = {}
    for element in manifest.elements:
        if str(getattr(element, "track", "") or "").strip() != "private":
            continue
        element_id = str(element.id)
        gt_type = getattr(element, "groundtruth_type", None)
        private[element_id] = str(getattr(gt_type, "value", gt_type) or "soccer_action")
    logger.info(
        "%sManifest source=%s block=%d private_elements=%s",
        LOG_PREFIX,
        index_url,
        block,
        sorted(private),
    )
    return private


def _winner_target(entry: dict[str, Any]) -> tuple[str, int] | None:
    hotkey = str(entry.get("winner_hotkey") or "").strip()
    if not hotkey:
        return None

    commit_blocks: list[int] = []
    for row in [entry, entry.get("winner_1"), *(entry.get("top_3_official") or [])]:
        if not isinstance(row, dict):
            continue
        row_hotkey = str(row.get("hotkey") or row.get("winner_hotkey") or "").strip()
        if row_hotkey != hotkey:
            continue
        raw_block = row.get("commit_block", row.get("winner_commit_block"))
        try:
            commit_blocks.append(int(raw_block))
        except (TypeError, ValueError):
            continue
    if not commit_blocks:
        return None
    return hotkey, max(commit_blocks)


def _candidate_keys(
    keys: list[str], element_id: str, hotkey: str, commit_block: int
) -> list[str]:
    safe_element_id = element_id.replace("/", "_")
    matches: list[str] = []
    for key in keys:
        key_element, key_hotkey, key_commit = extract_element_miner_commit_from_key(key)
        if (key_element, key_hotkey, key_commit) != (
            safe_element_id,
            hotkey,
            int(commit_block),
        ):
            continue
        if "/evaluation/" not in key:
            continue
        matches.append(key)
    matches.sort(key=lambda key: extract_block_from_key(key) or -1)
    if len(matches) <= RECENT_CHALLENGES_EXCLUDED:
        return []
    candidates = matches[:-RECENT_CHALLENGES_EXCLUDED]
    random.shuffle(candidates)
    return candidates


def _matching_payload(
    lines: list[dict[str, Any]], element_id: str, hotkey: str, commit_block: int
) -> dict[str, Any] | None:
    for line in lines:
        payload = line.get("payload") if isinstance(line, dict) else None
        if not isinstance(payload, dict) or payload.get("lane") != "private":
            continue
        telemetry = payload.get("telemetry") or {}
        miner = telemetry.get("miner") or {}
        commitment = miner.get("commitment") or {}
        try:
            payload_commit = int(commitment.get("commit_block"))
        except (TypeError, ValueError):
            continue
        if (
            str(payload.get("element_id") or "") == element_id
            and str(miner.get("hotkey") or "") == hotkey
            and payload_commit == int(commit_block)
        ):
            return payload
    return None


async def _select_private_shard(
    index_url: str,
    index_keys: list[str],
    element_id: str,
    hotkey: str,
    commit_block: int,
) -> dict[str, Any]:
    candidates = _candidate_keys(index_keys, element_id, hotkey, commit_block)
    if not candidates:
        raise RuntimeError(
            f"No eligible shard after excluding the latest {RECENT_CHALLENGES_EXCLUDED}"
        )
    for key in candidates:
        payload = _matching_payload(
            await fetch_shard_lines(index_url, key), element_id, hotkey, commit_block
        )
        if payload is not None:
            return payload
    raise RuntimeError("Eligible score shards contained no matching private payload")


def _shard_references(payload: dict[str, Any]) -> tuple[str, str]:
    telemetry = payload.get("telemetry") or {}
    run = telemetry.get("run") or {}
    response_key = str(run.get("responses_key") or run.get("response_key") or "").strip()
    task_id = str(
        telemetry.get("api_task_id")
        or telemetry.get("task_id")
        or telemetry.get("challenge_id")
        or ""
    ).strip()
    if not response_key:
        raise RuntimeError("Selected shard has no responses_key")
    if not task_id:
        raise RuntimeError("Selected shard has no task id")
    return response_key, task_id


async def _fetch_raw_ground_truth(task_id: str, element_id: str, keypair) -> Any:
    settings = get_settings()
    api_url = (settings.PRIVATE_GT_API_URL or settings.SCOREVISION_API).rstrip("/")
    if not api_url:
        raise RuntimeError("Neither PRIVATE_GT_API_URL nor SCOREVISION_API is configured")
    params = build_validator_query_params(keypair)
    params["element_id"] = element_id
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{api_url}/api/tasks/{int(task_id)}/ground-truth", params=params
        )
        response.raise_for_status()
        return response.json()


async def _store_export(trigger_block: int, payload: dict[str, Any]) -> str:
    config = central_r2_config(get_settings())
    if not config.bucket:
        raise RuntimeError("Public score R2 bucket is not configured")
    factory = _client_factory(config, "Public score R2 credentials are not configured")
    index_key = _index_key()
    await ensure_index_exists(
        client_factory=factory, bucket=config.bucket, index_key=index_key
    )
    key = _run_key(trigger_block)
    async with factory() as client:
        await client.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=dumps(payload, separators=(",", ":")),
            ContentType="application/json",
        )
    await add_index_key_if_new(
        client_factory=factory,
        bucket=config.bucket,
        key=key,
        index_key=index_key,
    )
    return key


async def run_private_audit_export_once(
    trigger_block: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    if trigger_block is None:
        subtensor = await get_subtensor()
        trigger_block = int(await subtensor.get_current_block())

    winners_block, snapshot = await load_latest_winners_snapshot()
    private_elements = await _private_manifest_elements(winners_block or trigger_block)
    score_index_url = settings.PRIVATE_R2_PUBLIC_INDEX_URL
    if not score_index_url:
        raise RuntimeError("PRIVATE_R2_PUBLIC_INDEX_URL is not configured")
    score_keys = await _fetch_index_keys(score_index_url)

    response_config = _private_responses_config()
    keypair = load_hotkey_keypair(
        settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT
    )
    result: dict[str, Any] = {}
    failures: list[str] = []
    winners = snapshot.get("winners") or {}
    for element_id, groundtruth_type in private_elements.items():
        entry = winners.get(element_id)
        target = _winner_target(entry) if isinstance(entry, dict) else None
        if target is None:
            message = f"{element_id}: winner or commit block missing"
            failures.append(message)
            logger.warning("%s%s", LOG_PREFIX, message)
            continue
        hotkey, commit_block = target
        try:
            shard = await _select_private_shard(
                score_index_url, score_keys, element_id, hotkey, commit_block
            )
            response_key, task_id = _shard_references(shard)
            miner_response, groundtruth = await asyncio.gather(
                _load_json_object(
                    response_config,
                    response_key,
                    label="Private responses R2",
                ),
                _fetch_raw_ground_truth(task_id, element_id, keypair),
            )
            result[element_id] = {
                "miner_response": miner_response,
                "groundtruth": groundtruth,
            }
            logger.info(
                "%sSelected element=%s hotkey=%s commit_block=%d task_id=%s groundtruth_type=%s",
                LOG_PREFIX,
                element_id,
                hotkey,
                commit_block,
                task_id,
                groundtruth_type,
            )
        except Exception as exc:
            failures.append(f"{element_id}: {exc}")
            logger.warning("%sFailed %s: %s", LOG_PREFIX, element_id, exc)

    if failures:
        raise RuntimeError(
            "Private audit export is incomplete; nothing was published: "
            + "; ".join(failures)
        )
    if not result:
        raise RuntimeError("Manifest contains no private elements to export")
    key = await _store_export(trigger_block, result)
    logger.info("%sStored %s with %d elements", LOG_PREFIX, key, len(result))
    return result


async def _last_export_block() -> int | None:
    config = central_r2_config(get_settings())
    if not config.bucket:
        return None
    try:
        data = await _load_json_object(config, _index_key(), label="Public score R2")
    except Exception as exc:
        logger.info("%sNo previous export index available: %s", LOG_PREFIX, exc)
        return None
    blocks = [_block_from_key(key) for key in data] if isinstance(data, list) else []
    valid = [block for block in blocks if block is not None]
    return max(valid) if valid else None


def _due_trigger_block(
    current_block: int,
    last_trigger_block: int | None,
    interval: int,
) -> int | None:
    interval = max(1, int(interval))
    current_block = int(current_block)
    if last_trigger_block is None:
        return current_block if current_block % interval == 0 else None
    if current_block < last_trigger_block + interval:
        return None
    return last_trigger_block + ((current_block - last_trigger_block) // interval) * interval


async def private_audit_export_loop() -> None:
    settings = get_settings()
    last_trigger = await _last_export_block()
    logger.info(
        "%sStarting interval=%d blocks last_trigger=%s",
        LOG_PREFIX,
        settings.PRIVATE_AUDIT_EXPORT_INTERVAL_BLOCKS,
        last_trigger,
    )
    previous_block: int | None = None
    while not shutdown_event.is_set():
        try:
            subtensor = await get_subtensor()
            block = int(await subtensor.get_current_block())
            due = _due_trigger_block(
                block, last_trigger, settings.PRIVATE_AUDIT_EXPORT_INTERVAL_BLOCKS
            )
            if last_trigger is None and previous_block is not None:
                interval = max(1, settings.PRIVATE_AUDIT_EXPORT_INTERVAL_BLOCKS)
                boundary = ((previous_block // interval) + 1) * interval
                if previous_block < boundary <= block:
                    due = boundary
            previous_block = block
            if due is not None:
                await run_private_audit_export_once(due)
                last_trigger = due
        except Exception as exc:
            logger.warning("%sLoop error: %s", LOG_PREFIX, exc)
            reset_subtensor()
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=max(20, settings.PRIVATE_AUDIT_EXPORT_POLL_INTERVAL_S),
            )
        except asyncio.TimeoutError:
            pass


def setup_shutdown_handler() -> None:
    loop = asyncio.get_running_loop()

    def stop() -> None:
        logger.warning("%sShutdown requested", LOG_PREFIX)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop)
