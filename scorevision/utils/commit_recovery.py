from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from json import dumps
from time import monotonic
from typing import Iterable, Sequence
from urllib.parse import urlparse

from scorevision.utils.r2_public import (
    extract_block_from_key,
    extract_element_miner_commit_from_key,
    fetch_json_from_url,
    fetch_shard_lines,
    normalize_index_url,
)


RECOVERY_COMMIT_ROLE = "miner_recover"
RECOVERY_CACHE_TTL_S = 300.0
MAX_RECOVERY_MISSED_TRIGGERS = 6
RECOVERY_ACTIVATION_BLOCK = 9_504_379


def is_timely_recovery(
    shard_block: int,
    recovery_block: int,
    observed_trigger_blocks: Sequence[int],
    *,
    activation_block: int | None = None,
) -> bool:
    """Grandfather recoveries before activation; then count indexed triggers."""
    if shard_block >= recovery_block:
        return False
    if activation_block is None:
        activation_block = RECOVERY_ACTIVATION_BLOCK
    if recovery_block < activation_block:
        return True
    return (
        bisect_left(observed_trigger_blocks, recovery_block)
        - bisect_right(observed_trigger_blocks, shard_block)
        <= MAX_RECOVERY_MISSED_TRIGGERS
    )


@dataclass(frozen=True)
class RecoveredCommitment:
    model: str
    revision: str
    slug: str
    chute_id: str
    element_id: str
    commit_block: int
    shard_block: int
    shard_key: str
    observed_trigger_blocks: tuple[int, ...] = ()

    def as_miner_commitment(self, hotkey: str) -> dict:
        return {
            "role": "miner",
            "model": self.model,
            "revision": self.revision,
            "slug": self.slug,
            "chute_id": self.chute_id,
            "hotkey": hotkey,
            "element_id": self.element_id,
        }


_RECOVERY_CACHE: dict[
    tuple[str, str], tuple[RecoveredCommitment, float, int | None]
] = {}


def is_recovery_commit(
    obj: object,
    element_id: str | None = None,
    *,
    hotkey: str | None = None,
) -> bool:
    if not isinstance(obj, dict) or obj.get("role") != RECOVERY_COMMIT_ROLE:
        return False
    committed_element = str(obj.get("element_id") or "").strip()
    if not committed_element:
        return False
    if element_id is not None and committed_element != str(element_id).strip():
        return False
    payload_hotkey = str(obj.get("hotkey") or "").strip()
    return hotkey is None or not payload_hotkey or payload_hotkey == str(hotkey).strip()


def _safe_element_id(element_id: str) -> str:
    return str(element_id).strip().replace("/", "_")


def _active_index_url(index_url: str) -> str | None:
    return normalize_index_url(index_url)


def _join_key(index_url: str, key: str) -> str:
    if key.startswith("http://") or key.startswith("https://"):
        return key
    parsed = urlparse(index_url)
    if key.startswith("/"):
        return f"{parsed.scheme}://{parsed.netloc}{key}"
    if key.startswith("manako/"):
        return f"{parsed.scheme}://{parsed.netloc}/{key}"
    return index_url.rsplit("/", 1)[0] + "/" + key


def _index_keys(data: object, index_url: str) -> list[str]:
    raw_keys: list[str] = []
    if isinstance(data, list):
        raw_keys = [key for key in data if isinstance(key, str)]
    elif isinstance(data, dict) and isinstance(data.get("entries"), list):
        raw_keys = [
            entry["path"]
            for entry in data["entries"]
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        ]
    return [_join_key(index_url, key) for key in raw_keys]


def extract_recovered_commitment(
    line: dict,
    *,
    shard_key: str,
    hotkey: str,
    element_id: str,
) -> RecoveredCommitment | None:
    # Imported lazily to keep bittensor_helpers -> commit_recovery free of an
    # import cycle (cloudflare_helpers also uses bittensor_helpers).
    from scorevision.utils.cloudflare_helpers import _verify_signature

    payload = line.get("payload")
    if not isinstance(payload, dict):
        return None

    signer = str(line.get("hotkey") or "").strip()
    signature = str(line.get("signature") or "").strip()
    payload_text = dumps(payload, sort_keys=True, separators=(",", ":"))
    if not signer or not signature or not _verify_signature(signer, payload_text, signature):
        return None

    key_element, key_hotkey, key_commit_block = extract_element_miner_commit_from_key(
        shard_key
    )
    if (
        key_element != _safe_element_id(element_id)
        or key_hotkey != hotkey
        or key_commit_block < 0
    ):
        return None
    if str(payload.get("element_id") or "").strip() != str(element_id).strip():
        return None
    if str(payload.get("lane") or "public").strip().lower() != "public":
        return None

    telemetry = payload.get("telemetry")
    miner = telemetry.get("miner") if isinstance(telemetry, dict) else None
    if not isinstance(miner, dict) or str(miner.get("hotkey") or "").strip() != hotkey:
        return None
    commitment = miner.get("commitment")
    if not isinstance(commitment, dict):
        return None

    committed_element = str(commitment.get("element_id") or "").strip()
    if committed_element != str(element_id).strip():
        return None
    try:
        committed_block = int(commitment.get("commit_block"))
    except (TypeError, ValueError):
        return None
    if committed_block != key_commit_block:
        return None

    values = {
        "model": miner.get("model") or commitment.get("model"),
        "revision": miner.get("revision") or commitment.get("revision"),
        "slug": miner.get("slug") or commitment.get("chute_slug"),
        "chute_id": miner.get("chute_id") or commitment.get("chute_id"),
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        return None

    shard_block = extract_block_from_key(shard_key)
    if shard_block is None:
        return None
    return RecoveredCommitment(
        model=str(values["model"]).strip(),
        revision=str(values["revision"]).strip(),
        slug=str(values["slug"]).strip(),
        chute_id=str(values["chute_id"]).strip(),
        element_id=str(element_id).strip(),
        commit_block=committed_block,
        shard_block=shard_block,
        shard_key=shard_key,
    )


async def recover_commitments_from_shards(
    requests: Iterable[tuple[str, str]],
    validator_indexes: dict[str, str],
    *,
    concurrency: int = 8,
    recovery_blocks: dict[tuple[str, str], int] | None = None,
    index_keys_by_url: dict[str, list[str]] | None = None,
) -> dict[tuple[str, str], RecoveredCommitment]:
    wanted = {
        (str(hotkey).strip(), str(element_id).strip())
        for hotkey, element_id in requests
        if str(hotkey).strip() and str(element_id).strip()
    }
    if not wanted or not validator_indexes:
        return {}

    now = monotonic()
    cached_by_request = {
        request: cached
        for request in wanted
        if (entry := _RECOVERY_CACHE.get(request)) is not None
        and now - entry[1] < RECOVERY_CACHE_TTL_S
        and entry[2]
        == (recovery_blocks.get(request) if recovery_blocks is not None else None)
        for cached in (entry[0],)
    }
    if recovery_blocks is None:
        wanted.difference_update(cached_by_request)
        if not wanted:
            return cached_by_request
    recovered_by_request = dict(cached_by_request) if recovery_blocks is None else {}

    candidate_keys: dict[tuple[str, str], set[str]] = {request: set() for request in wanted}
    requests_by_path: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for hotkey, element_id in wanted:
        requests_by_path.setdefault(
            (_safe_element_id(element_id), hotkey),
            [],
        ).append((hotkey, element_id))
    indexed_trigger_blocks: dict[str, set[int]] = {
        _safe_element_id(element_id): set() for _, element_id in wanted
    }
    for raw_index_url in validator_indexes.values():
        index_url = _active_index_url(raw_index_url)
        if not index_url:
            continue
        if index_keys_by_url is not None and index_url in index_keys_by_url:
            keys = index_keys_by_url[index_url]
        else:
            data = await fetch_json_from_url(index_url)
            keys = _index_keys(data, index_url)
        for key in keys:
            key_element, key_hotkey, key_commit_block = (
                extract_element_miner_commit_from_key(key)
            )
            if key_commit_block < 0:
                continue
            if "/evaluation/" not in urlparse(key).path:
                continue
            shard_block = extract_block_from_key(key)
            if shard_block is None:
                continue
            if key_element in indexed_trigger_blocks:
                indexed_trigger_blocks[key_element].add(shard_block)
            for request in requests_by_path.get((key_element, key_hotkey), []):
                if (
                    recovery_blocks is None
                    or shard_block < recovery_blocks.get(request, 0)
                ):
                    candidate_keys[request].add(key)

    triggers_by_element = {
        element_id: tuple(sorted(blocks))
        for element_id, blocks in indexed_trigger_blocks.items()
    }

    if recovery_blocks is not None:
        for request, cached in cached_by_request.items():
            keys = candidate_keys[request]
            if cached.shard_key not in keys or any(
                ((extract_block_from_key(key) or -1), key)
                > (cached.shard_block, cached.shard_key)
                for key in keys
            ):
                continue
            recovered_by_request[request] = replace(
                cached,
                observed_trigger_blocks=triggers_by_element[_safe_element_id(request[1])],
            )
            candidate_keys.pop(request)

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _fetch_candidate(
        request: tuple[str, str], key: str
    ) -> RecoveredCommitment | None:
        async with semaphore:
            lines = await fetch_shard_lines(key, key)
        hotkey, element_id = request
        recovered = [
            item
            for line in lines
            if isinstance(line, dict)
            and (
                item := extract_recovered_commitment(
                    line,
                    shard_key=key,
                    hotkey=hotkey,
                    element_id=element_id,
                )
            )
            is not None
        ]
        return max(recovered, key=lambda item: item.shard_block, default=None)

    async def _recover_request(
        request: tuple[str, str], keys: set[str]
    ) -> RecoveredCommitment | None:
        ordered_keys = sorted(
            keys,
            key=lambda key: (extract_block_from_key(key) or -1, key),
            reverse=True,
        )
        for key in ordered_keys:
            recovered = await _fetch_candidate(request, key)
            if recovered is not None:
                return recovered
        return None

    results = await asyncio.gather(
        *[
            _recover_request(request, keys)
            for request, keys in candidate_keys.items()
        ],
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException) or result is None:
            continue
        _element, recovered_hotkey, _commit = extract_element_miner_commit_from_key(
            result.shard_key
        )
        request = (recovered_hotkey, result.element_id)
        previous = recovered_by_request.get(request)
        if previous is None or result.shard_block > previous.shard_block:
            result = replace(
                result,
                observed_trigger_blocks=triggers_by_element[
                    _safe_element_id(result.element_id)
                ],
            )
            recovered_by_request[request] = result
            _RECOVERY_CACHE[request] = (
                result,
                monotonic(),
                recovery_blocks.get(request) if recovery_blocks is not None else None,
            )
    return recovered_by_request
