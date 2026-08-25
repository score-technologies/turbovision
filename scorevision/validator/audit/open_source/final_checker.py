"""Final checker: the only place where a compliance verdict is decided.

Runs on the owner profile. The latency loop merely measures and signs what it saw,
one immutable run at a time; every piece of state that a verdict depends on — the
running streaks, the cursor, the fails list — lives here, in a bucket the latency
machine has no credentials for. Stealing the conformity keys therefore buys nothing:
there is no counter to inflate and none to erase.
"""

from __future__ import annotations

import asyncio
from json import dumps, loads
from logging import getLogger
from re import compile as re_compile
from time import time
from typing import Any

from scorevision.utils.r2 import R2Config, central_r2_config, create_s3_client, is_configured
from scorevision.utils.r2_public import extract_base_url, fetch_json_from_url
from scorevision.utils.run_signing import verify_run_payload
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

LATENCY_BREACH_STATUSES = ("PENDING_LATENCY", "FAIL_LATENCY")
# SKIP and FAIL_RUNTIME describe our own inability to measure, never the miner's
# behaviour, so they are not observations and must not influence a verdict.
OBSERVED_STATUSES = ("PASS", "PENDING_LATENCY", "FAIL_LATENCY", "FAIL_OUTPUT")

MAX_CONCURRENT_FETCHES = 8
# Anyone holding the conformity key can create objects under the runs prefix — the
# lock only forbids overwriting and deleting. Only keys the checker could have
# produced are worth fetching, and no cycle should be made unbounded by junk.
RUN_KEY_PATTERN = re_compile(r"^\d{9}\.json$")
MAX_RUNS_PER_CYCLE = 200


def _is_run_key(key: str, prefix: str) -> bool:
    return key.startswith(prefix) and bool(RUN_KEY_PATTERN.match(key[len(prefix) :]))


# ---------------------------------------------------------------- reading runs


def _public_base() -> str:
    base = (get_settings().CHECKER_R2_BUCKET_PUBLIC_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("CHECKER_R2_BUCKET_PUBLIC_URL not set: cannot read compliance runs")
    return extract_base_url(base).rstrip("/")


def _runs_prefix() -> str:
    return f"{get_settings().CHECKER_R2_RESULTS_PREFIX.strip().strip('/')}/runs/"


def _url_for_key(key: str) -> str:
    return f"{_public_base()}/{str(key).strip().lstrip('/')}"


def _read_config() -> R2Config | None:
    """Read-only credentials for the conformity bucket, if provisioned."""
    settings = get_settings()
    cfg = R2Config(
        bucket=(settings.CHECKER_R2_BUCKET or "").strip(),
        account_id=settings.CHECKER_R2_ACCOUNT_ID.get_secret_value(),
        access_key_id=settings.CHECKER_R2_READ_ACCESS_KEY_ID.get_secret_value(),
        secret_access_key=settings.CHECKER_R2_READ_SECRET_ACCESS_KEY.get_secret_value(),
        concurrency=settings.CHECKER_R2_CONCURRENCY,
    )
    return cfg if is_configured(cfg, require_bucket=True) else None


async def _list_run_keys(cfg: R2Config, after: str | None) -> list[str]:
    """Enumerate runs from the bucket itself.

    The index is a mutable object: an entry can be removed before we ever read it,
    and the run it points at then goes unobserved. A listing cannot be edited, so
    discovering runs this way removes the problem instead of detecting it.
    """
    prefix = _runs_prefix()
    keys: list[str] = []
    token: str | None = None
    async with create_s3_client(cfg, error_message="Conformity read credentials not set") as client:
        while True:
            params: dict[str, Any] = {"Bucket": cfg.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                params["ContinuationToken"] = token
            elif after:
                params["StartAfter"] = after
            response = await client.list_objects_v2(**params)
            for item in response.get("Contents") or []:
                key = str(item.get("Key") or "")
                if _is_run_key(key, prefix):
                    keys.append(key)
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if not token:
                break
    return sorted(keys)


async def _fetch_index() -> list[str]:
    data = await fetch_json_from_url(_url_for_key(f"{_runs_prefix()}index.json"))
    if not isinstance(data, list):
        logger.error("[final-checker] runs index unavailable or malformed")
        return []
    prefix = _runs_prefix()
    return [k for k in data if isinstance(k, str) and _is_run_key(k, prefix)]


async def _fetch_payloads(keys: list[str]) -> list[tuple[str, dict[str, Any] | None]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def fetch(key: str) -> tuple[str, dict[str, Any] | None]:
        async with semaphore:
            payload = await fetch_json_from_url(_url_for_key(key))
        return key, payload if isinstance(payload, dict) else None

    return list(await asyncio.gather(*(fetch(k) for k in keys)))


def _is_authentic(key: str, payload: dict[str, Any]) -> bool:
    expected_hotkey = (get_settings().LATENCY_LOOP_HOTKEY or "").strip()
    if not expected_hotkey:
        return True
    if verify_run_payload(payload, run_key=key, expected_hotkey=expected_hotkey):
        return True
    logger.warning("[final-checker] run rejected: signature mismatch key=%s", key)
    return False


# ------------------------------------------------------------------- verdicts


def _tuple_id(hotkey: str, element_id: str, commit_block: int) -> str:
    return f"{hotkey}|{element_id}|{commit_block}"


def _is_latency_breach(row: dict[str, Any]) -> bool:
    """A breach is a measurement, not a label: p95 must really exceed the bar."""
    if str(row.get("status") or "") not in LATENCY_BREACH_STATUSES:
        return False
    try:
        return float(row.get("p95_latency_ms")) > float(row.get("effective_latency_threshold_ms"))
    except (TypeError, ValueError):
        return False


def apply_run(
    tuples: dict[str, dict[str, Any]],
    run_key: str,
    payload: dict[str, Any],
    *,
    streak_threshold: int,
) -> dict[str, dict[str, Any]]:
    """Fold one run into the running verdicts.

    Only tuples the run covers are touched: one that stops being targeted keeps its
    verdict rather than quietly expiring, which would let a banned miner come back.
    """
    threshold = max(1, int(streak_threshold))
    try:
        ts = float(payload.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0

    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status not in OBSERVED_STATUSES:
            continue
        try:
            hotkey = str(row["hotkey"])
            element_id = str(row["element_id"])
            commit_block = int(row["commit_block"])
        except Exception:
            continue
        tuple_id = _tuple_id(hotkey, element_id, commit_block)

        if status == "FAIL_OUTPUT":
            tuples[tuple_id] = {
                "hotkey": hotkey,
                "element_id": element_id,
                "commit_block": commit_block,
                "verdict": "FAIL_OUTPUT",
                "run_keys": [run_key],
                "first_seen": ts,
                "last_seen": ts,
            }
            continue

        if not _is_latency_breach(row):
            tuples.pop(tuple_id, None)
            continue

        entry = tuples.get(tuple_id)
        if entry is None or entry.get("verdict") == "FAIL_OUTPUT":
            entry = {
                "hotkey": hotkey,
                "element_id": element_id,
                "commit_block": commit_block,
                "run_keys": [],
                "first_seen": ts,
            }
            tuples[tuple_id] = entry
        run_keys = [k for k in entry.get("run_keys") or [] if k != run_key]
        run_keys.append(run_key)
        entry["run_keys"] = run_keys[-threshold:]
        entry["last_seen"] = ts
        entry["verdict"] = "FAIL_LATENCY" if len(run_keys) >= threshold else "PENDING_LATENCY"

    return tuples


def failing_rows(
    tuples: dict[str, dict[str, Any]],
    *,
    streak_threshold: int,
) -> list[dict[str, Any]]:
    """Settled verdicts only; a streak still building stays internal."""
    threshold = max(1, int(streak_threshold))
    rows: list[dict[str, Any]] = []
    for entry in tuples.values():
        verdict = entry.get("verdict")
        run_keys = list(entry.get("run_keys") or [])
        if verdict == "FAIL_LATENCY" and len(run_keys) < threshold:
            continue
        if verdict not in ("FAIL_OUTPUT", "FAIL_LATENCY"):
            continue
        rows.append(
            {
                "hotkey": entry["hotkey"],
                "element_id": entry["element_id"],
                "commit_block": entry["commit_block"],
                "first_seen": entry.get("first_seen"),
                "last_seen": entry.get("last_seen"),
                "latest_status": verdict,
                "latest_run_key": run_keys[-1] if run_keys else None,
                "evidence_run_keys": run_keys,
            }
        )
    return sorted(rows, key=lambda r: (r["hotkey"], r["element_id"], r["commit_block"]))


# --------------------------------------------------------------- owner bucket


def _owner_config() -> R2Config:
    cfg = central_r2_config(get_settings())
    if not is_configured(cfg, require_bucket=True):
        raise RuntimeError("owner R2 credentials not set: cannot read or publish the fails list")
    return cfg


async def _get_owner_json(key: str) -> Any | None:
    cfg = _owner_config()
    async with create_s3_client(cfg, error_message="Owner R2 credentials not set") as client:
        try:
            obj = await client.get_object(Bucket=cfg.bucket, Key=key)
            return loads((await obj["Body"].read()).decode())
        except Exception:
            return None


async def _put_owner_json(key: str, payload: Any) -> None:
    cfg = _owner_config()
    async with create_s3_client(cfg, error_message="Owner R2 credentials not set") as client:
        await client.put_object(
            Bucket=cfg.bucket,
            Key=key,
            Body=dumps(payload, separators=(",", ":")),
            ContentType="application/json",
        )


# ------------------------------------------------------------------ the cycle


def chain_is_continuous(payload: dict[str, Any], cursor: str | None) -> bool:
    """Does this run follow the one we last processed?

    The runs index is mutable, so an entry can be removed before we ever read it.
    Each run names its predecessor, which turns that removal into a visible break
    instead of a silently missing observation.
    """
    previous = str(payload.get("prev_run_key") or "").strip()
    if not cursor or not previous:
        return True
    return previous == cursor


def _new_keys(index: list[str], cursor: str | None) -> list[str]:
    """Strictly increasing keys only: re-inserting an old key must not replay it."""
    if not cursor:
        return index
    return [k for k in index if k > cursor]


async def run_final_check_once() -> dict[str, Any]:
    settings = get_settings()
    threshold = settings.CHECKER_LATENCY_FAIL_STREAK_THRESHOLD
    state_key = settings.FINAL_CHECKER_STATE_KEY.strip()
    output_key = settings.FINAL_CHECKER_OUTPUT_KEY.strip()

    state = await _get_owner_json(state_key)
    if not isinstance(state, dict):
        logger.warning("[final-checker] no usable state at %s, starting from scratch", state_key)
        state = {}
    tuples: dict[str, dict[str, Any]] = state.get("tuples") if isinstance(state.get("tuples"), dict) else {}
    cursor = state.get("last_run_key")

    read_cfg = _read_config()
    if read_cfg is not None:
        pending = await _list_run_keys(read_cfg, cursor)
        discovery = "listing"
    else:
        # No read token: fall back to the mutable index, which the chain check
        # below can only report on after the fact.
        index = await _fetch_index()
        if not index:
            return {"published": False, "reason": "empty_index"}
        pending = _new_keys(index, cursor)
        discovery = "index"

    if not pending:
        logger.info("[final-checker] no new run since %s", cursor)
        return {"published": False, "reason": "no_new_run"}

    if len(pending) > MAX_RUNS_PER_CYCLE:
        logger.warning(
            "[final-checker] %d runs pending, processing the first %d this cycle",
            len(pending),
            MAX_RUNS_PER_CYCLE,
        )
        pending = pending[:MAX_RUNS_PER_CYCLE]

    applied = 0
    rejected = 0
    for key, payload in await _fetch_payloads(pending):
        if payload is None:
            logger.warning("[final-checker] run unreadable, stopping at key=%s", key)
            break
        if not _is_authentic(key, payload):
            rejected += 1
            cursor = key
            continue
        if not chain_is_continuous(payload, cursor):
            logger.error(
                "[final-checker] run chain broken: %s claims to follow %s but we last saw %s "
                "(a run may have been hidden from the index)",
                key,
                payload.get("prev_run_key"),
                cursor,
            )
        tuples = apply_run(tuples, key, payload, streak_threshold=threshold)
        applied += 1
        cursor = key

    rows = failing_rows(tuples, streak_threshold=threshold)
    await _put_owner_json(output_key, rows)
    await _put_owner_json(
        state_key,
        {"last_run_key": cursor, "tuples": tuples, "rows": len(rows), "updated_at": time()},
    )

    logger.info(
        "[final-checker] done discovery=%s applied=%d rejected=%d tracked=%d banned=%d cursor=%s",
        discovery,
        applied,
        rejected,
        len(tuples),
        len(rows),
        cursor,
    )
    return {
        "applied": applied,
        "rejected": rejected,
        "tracked": len(tuples),
        "banned": len(rows),
        "published": True,
    }


async def final_checker_loop() -> None:
    settings = get_settings()
    while True:
        try:
            await run_final_check_once()
        except Exception as e:
            logger.warning("[final-checker] loop error: %s", e)
        await asyncio.sleep(max(30, settings.FINAL_CHECKER_POLL_INTERVAL_S))
