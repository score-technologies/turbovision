from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import aiohttp

logger = getLogger(__name__)

DEFAULT_INACTIVE_MINERS_URL = "https://turbo.scoredata.me/manako/inactive_miners.json"
_FETCH_CACHE: dict[str, tuple[set["InactiveMinerTuple"], float]] = {}
_FETCH_TTL_S = 300.0


@dataclass(frozen=True)
class InactiveMinerTuple:
    hotkey: str
    element_id: str
    commit_block: int


def normalize_inactive_miner_tuple(
    hotkey: Any,
    element_id: Any,
    commit_block: Any,
) -> InactiveMinerTuple | None:
    try:
        hk = str(hotkey or "").strip()
        eid = str(element_id or "").strip()
        cb = int(commit_block)
    except Exception:
        return None
    if not hk or not eid or cb < 0:
        return None
    return InactiveMinerTuple(hotkey=hk, element_id=eid, commit_block=cb)


def parse_inactive_miner_tuples(data: Any) -> set[InactiveMinerTuple]:
    rows = data if isinstance(data, list) else []
    parsed: set[InactiveMinerTuple] = set()
    for row in rows:
        item = None
        if isinstance(row, dict):
            item = normalize_inactive_miner_tuple(
                row.get("hotkey"),
                row.get("element_id"),
                row.get("commit_block"),
            )
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            item = normalize_inactive_miner_tuple(row[0], row[1], row[2])
        if item is not None:
            parsed.add(item)
    return parsed


def is_inactive_miner_tuple(
    inactive_miners: set[InactiveMinerTuple] | None,
    *,
    hotkey: str | None,
    element_id: str | None,
    commit_block: int | str | None,
) -> bool:
    if not inactive_miners:
        return False
    item = normalize_inactive_miner_tuple(hotkey, element_id, commit_block)
    return item in inactive_miners if item is not None else False


async def fetch_inactive_miner_tuples(
    url: str = DEFAULT_INACTIVE_MINERS_URL,
    *,
    timeout_s: float = 10.0,
    use_cache: bool = True,
) -> set[InactiveMinerTuple]:
    url = str(url or "").strip()
    if not url:
        logger.warning("[inactive-miners] URL is empty")
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
                    logger.warning("[inactive-miners] GET %s -> %s", url, response.status)
                    return set(cached[0]) if cached else set()
                data = await response.json()
    except Exception as e:
        logger.warning("[inactive-miners] failed to fetch %s: %s", url, e)
        return set(cached[0]) if cached else set()

    inactive_miners = parse_inactive_miner_tuples(data)
    _FETCH_CACHE[url] = (set(inactive_miners), now)
    logger.info("[inactive-miners] loaded %d tuple(s)", len(inactive_miners))
    return inactive_miners
