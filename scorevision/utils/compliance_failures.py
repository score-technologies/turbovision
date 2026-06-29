from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import aiohttp

from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

DEFAULT_FAILING_TUPLES_URL = "https://manako.scoredata.me/manako/compliances/failing_tuples.json"
_FETCH_CACHE: dict[str, tuple[set["ComplianceFailureTuple"], float]] = {}
_FETCH_TTL_S = 300.0


@dataclass(frozen=True)
class ComplianceFailureTuple:
    hotkey: str
    element_id: str
    commit_block: int


def normalize_compliance_failure_tuple(
    hotkey: Any,
    element_id: Any,
    commit_block: Any,
) -> ComplianceFailureTuple | None:
    try:
        hk = str(hotkey or "").strip()
        eid = str(element_id or "").strip()
        cb = int(commit_block)
    except Exception:
        return None
    if not hk or not eid or cb < 0:
        return None
    return ComplianceFailureTuple(hotkey=hk, element_id=eid, commit_block=cb)


def parse_compliance_failure_tuples(data: Any) -> set[ComplianceFailureTuple]:
    rows: list[Any]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        raw_rows = data.get("tuples") or data.get("entries") or data.get("failures") or []
        rows = raw_rows if isinstance(raw_rows, list) else []
    else:
        rows = []

    parsed: set[ComplianceFailureTuple] = set()
    for row in rows:
        item = None
        if isinstance(row, dict):
            item = normalize_compliance_failure_tuple(
                row.get("hotkey"),
                row.get("element_id") or row.get("element"),
                row.get("commit_block", row.get("block")),
            )
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            item = normalize_compliance_failure_tuple(row[0], row[1], row[2])
        if item is not None:
            parsed.add(item)
    return parsed


def is_compliance_tuple_failed(
    failures: set[ComplianceFailureTuple] | None,
    *,
    hotkey: str | None,
    element_id: str | None,
    commit_block: int | str | None,
) -> bool:
    if not failures:
        return False
    item = normalize_compliance_failure_tuple(hotkey, element_id, commit_block)
    return item in failures if item is not None else False


async def fetch_compliance_failure_tuples(
    url: str | None = None,
    *,
    timeout_s: float = 10.0,
    use_cache: bool = True,
) -> set[ComplianceFailureTuple]:
    if url is None:
        settings = get_settings()
        url = (getattr(settings, "SCOREVISION_FAILING_TUPLES_URL", "") or "").strip()
    url = str(url or "").strip()
    if not url:
        return set()

    now = time.time()
    cached = _FETCH_CACHE.get(url)
    if use_cache and cached and (now - cached[1]) < _FETCH_TTL_S:
        return set(cached[0])

    try:
        timeout = aiohttp.ClientTimeout(total=float(timeout_s))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning("[compliance-failures] GET %s -> %s", url, response.status)
                    return set(cached[0]) if cached else set()
                data = await response.json()
    except Exception as e:
        logger.warning("[compliance-failures] failed to fetch %s: %s", url, e)
        return set(cached[0]) if cached else set()

    failures = parse_compliance_failure_tuples(data)
    _FETCH_CACHE[url] = (set(failures), now)
    logger.info("[compliance-failures] loaded %d failing tuple(s)", len(failures))
    return failures
