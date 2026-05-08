from __future__ import annotations
import asyncio
import os, json, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from logging import getLogger
from urllib.parse import urljoin, urlparse

import aiohttp
from huggingface_hub import HfApi
from bittensor import async_subtensor

from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    reset_subtensor,
    get_validator_indexes_from_chain,
)
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

HARDCODED_BLACKLIST_HOTKEYS: set[str] = {
    "5DvY7cxtAvUeA2Goq26LNyzqSfPyjfY9SUsD4bgJa5PMnVNa",
    "5CMaFwgm2rPka66iUcgAa2SpBPskk6KqAGWZeKVx8APLnqTZ",
    "5CfGbGvZz6YUUPT84ntoGHANy1ddk9xGJiaQZVEb9Qi57Foc",
    "5CG3NLAVRzs1uxqFHTjUaSESW4N7PLFYrRosjuh7JRKFoYmi",
    "5CqKodaU2F5atWnLBivA2TeQWoJ9jxfPwADQ2HAWfAQHNsZV",
    "5CmCgAKAW1B3YmBDeVJxxd2MbhMeC6n1esJsVFdb2qUvD55r",
    "5Fnx48i6A6KtN9a8pAxQDZDQo3Z49JeviaLkcJUbLB9yCM4Q",
}

REGISTRY_BYPASS_UIDS = {6}
REGISTRY_BYPASS_HOTKEYS = {"5FsREvyUXSZWYRqVyQLDdpYmZZPnkhZyW6HjooozKP1nQkwu"}


def is_registry_bypass(uid: int | None, hotkey: str | None) -> bool:
    if uid is None or not hotkey:
        return False
    hk = str(hotkey).strip()
    return uid in REGISTRY_BYPASS_UIDS and hk in REGISTRY_BYPASS_HOTKEYS


@dataclass
class Miner:
    uid: int
    hotkey: str
    model: Optional[str]
    revision: Optional[str]
    slug: Optional[str]
    chute_id: Optional[str]
    block: int
    element_id: Optional[str] = None
    registry_skip_reason: Optional[str] = None


# ------------------------- HF gating & revision checks ------------------------- #
_HF_MODEL_GATING_CACHE: Dict[str, Tuple[bool, float]] = {}
_HF_GATING_TTL = 300  # seconds
_HF_MODEL_SIZE_CACHE: Dict[Tuple[str, str], Tuple[Optional[float], float]] = {}
_HF_MODEL_SIZE_TTL = 300  # seconds
_HF_ONNX_ONLY_CACHE: Dict[Tuple[str, str], Tuple[Optional[bool], float]] = {}
_HF_ONNX_ONLY_TTL = 300  # seconds
_CHUTES_FETCH_RETRIES = max(1, int(os.getenv("SV_REGISTRY_CHUTES_RETRIES", "2")))
_CHUTES_FETCH_BACKOFF_S = float(os.getenv("SV_REGISTRY_CHUTES_RETRY_BACKOFF_S", "0.5"))
_REGISTRY_COMMIT_BACKFILL_ENABLE = str(
    os.getenv("SV_REGISTRY_COMMIT_BACKFILL_ENABLE", "true")
).strip().lower() in ("1", "true", "yes", "on")
_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT = os.getenv(
    "SV_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT",
    "wss://archive.chain.opentensor.ai:443",
).strip()
_REGISTRY_COMMIT_BACKFILL_MAX_HOPS = max(
    1, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_MAX_HOPS", "20"))
)
_REGISTRY_COMMIT_BACKFILL_CONCURRENCY = max(
    1, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_CONCURRENCY", "1"))
)
_REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK = max(
    0, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK", "0"))
)
_REGISTRY_BACKFILL_INDEX_TIMEOUT_S = float(
    os.getenv("SV_REGISTRY_BACKFILL_INDEX_TIMEOUT_S", "12")
)


async def _hf_is_gated(model_id: str) -> Optional[bool]:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status == 200:
                    data = await r.json()
                    gated = bool(data.get("gated", False))
                    logger.debug("[HF] model=%s gated=%s", model_id, gated)
                    return gated
                logger.debug("[HF] model=%s status=%s", model_id, r.status)
    except Exception as e:
        logger.debug("[HF] is_gated error for %s: %s", model_id, e)
    return None


def _hf_revision_accessible(model_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return True
    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
        logger.debug("[HF] model=%s revision=%s accessible", model_id, revision)
        return True
    except Exception as e:
        logger.debug(
            "[HF] model=%s revision=%s NOT accessible: %s", model_id, revision, e
        )
        return False


async def _hf_gated_or_inaccessible(
    model_id: Optional[str], revision: Optional[str]
) -> Optional[bool]:
    if not model_id:
        logger.debug("[HF] no model id → treat as not eligible")
        return True
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
        logger.debug("[HF] cache hit model=%s gated=%s", model_id, gated)
    else:
        gated = await _hf_is_gated(model_id)
        _HF_MODEL_GATING_CACHE[model_id] = (bool(gated) if gated is not None else False, now)
        logger.debug("[HF] cache set model=%s gated=%s", model_id, gated)

    if gated is True:
        logger.info("[HF] model=%s is gated", model_id)
        return True
    if not _hf_revision_accessible(model_id, revision):
        logger.info("[HF] model=%s revision inaccessible", model_id)
        return True
    return False


def _hf_repo_total_size_mb(model_id: str, revision: Optional[str]) -> Optional[float]:
    revision_key = str(revision or "")
    cache_key = (model_id, revision_key)
    now = time.time()
    cached = _HF_MODEL_SIZE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _HF_MODEL_SIZE_TTL:
        logger.debug(
            "[HF] size cache hit model=%s revision=%s size_mb=%s",
            model_id,
            revision_key,
            cached[0],
        )
        return cached[0]

    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        total_bytes = 0
        for node in api.list_repo_tree(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        ):
            size = getattr(node, "size", None)
            if isinstance(size, int) and size >= 0:
                total_bytes += size

        size_mb = total_bytes / (1024 * 1024)
        _HF_MODEL_SIZE_CACHE[cache_key] = (size_mb, now)
        logger.debug(
            "[HF] computed size model=%s revision=%s size_mb=%.2f",
            model_id,
            revision_key,
            size_mb,
        )
        return size_mb
    except Exception as e:
        logger.info(
            "[HF] failed to compute repo size model=%s revision=%s: %s",
            model_id,
            revision_key,
            e,
        )
        _HF_MODEL_SIZE_CACHE[cache_key] = (None, now)
        return None


def _hf_repo_has_only_onnx_models(
    model_id: str, revision: Optional[str]
) -> Optional[bool]:
    revision_key = str(revision or "")
    cache_key = (model_id, revision_key)
    now = time.time()
    cached = _HF_ONNX_ONLY_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _HF_ONNX_ONLY_TTL:
        logger.debug(
            "[HF] onnx cache hit model=%s revision=%s onnx_only=%s",
            model_id,
            revision_key,
            cached[0],
        )
        return cached[0]

    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        model_exts = (
            ".onnx",
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".ckpt",
            ".h5",
            ".keras",
            ".pb",
            ".tflite",
            ".msgpack",
            ".gguf",
        )

        has_onnx = False
        found_non_onnx_model = False

        for node in api.list_repo_tree(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        ):
            path = str(getattr(node, "path", "") or "").lower()
            if not path:
                continue
            if not path.endswith(model_exts):
                continue
            if path.endswith(".onnx"):
                has_onnx = True
                continue
            found_non_onnx_model = True
            break

        onnx_only = has_onnx and not found_non_onnx_model
        _HF_ONNX_ONLY_CACHE[cache_key] = (onnx_only, now)
        logger.debug(
            "[HF] onnx scan model=%s revision=%s has_onnx=%s non_onnx_model=%s",
            model_id,
            revision_key,
            has_onnx,
            found_non_onnx_model,
        )
        return onnx_only
    except Exception as e:
        logger.info(
            "[HF] failed to verify onnx-only model repo=%s revision=%s: %s",
            model_id,
            revision_key,
            e,
        )
        _HF_ONNX_ONLY_CACHE[cache_key] = (None, now)
        return None


# ------------------------------ Chutes helpers -------------------------------- #
async def _chutes_get_json(url: str, headers: Dict[str, str]) -> Optional[dict]:
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers) as r:
                if r.status != 200:
                    logger.debug("[Chutes] GET %s -> %s", url, r.status)
                    return None
                try:
                    data = await r.json()
                    logger.debug("[Chutes] GET %s -> ok", url)
                    return data
                except Exception as e:
                    logger.debug("[Chutes] JSON decode error for %s: %s", url, e)
                    return None
    except Exception as e:
        logger.info("[Chutes] GET %s failed: %s", url, e)
        return None


async def fetch_chute_info(chute_id: str) -> Optional[dict]:
    token = os.getenv("CHUTES_API_KEY", "")
    if not token or not chute_id:
        logger.debug("[Chutes] missing token or chute_id")
        return None
    url = f"https://api.chutes.ai/chutes/{chute_id}"
    headers = {"Authorization": token}

    for attempt in range(1, _CHUTES_FETCH_RETRIES + 1):
        data = await _chutes_get_json(url, headers=headers)
        if data:
            return data
        if attempt < _CHUTES_FETCH_RETRIES:
            delay_s = _CHUTES_FETCH_BACKOFF_S * (2 ** (attempt - 1))
            logger.info(
                "[Chutes] retrying chute lookup chute_id=%s attempt=%s/%s in %.2fs",
                chute_id,
                attempt + 1,
                _CHUTES_FETCH_RETRIES,
                delay_s,
            )
            await asyncio.sleep(delay_s)
    return None

def _pick_latest_miner_commit_for_element(arr, wanted_element_id: str | None):
    best_blk = None
    best_data = None
    best_obj = None

    for blk, data in arr:
        try:
            blk_i = int(blk)
        except Exception:
            continue

        try:
            obj = json.loads(data)
        except Exception:
            continue

        role = obj.get("role")
        if role != "miner":
            continue

        committed_eid = obj.get("element_id")
        committed_eid = str(committed_eid).strip() if committed_eid is not None else None

        if wanted_element_id is not None and committed_eid != wanted_element_id:
            continue

        if best_blk is None or blk_i > best_blk:
            best_blk = blk_i
            best_data = data
            best_obj = obj

    return best_blk, best_data, best_obj


async def _find_miner_commit_via_archive_backfill(
    st_archive,
    *,
    netuid: int,
    hotkey: str,
    initial_arr,
    wanted_element_id: str,
    max_hops: int,
    first_block: int,
) -> tuple[int | None, dict | None]:
    if len(initial_arr or []) < 10:
        return None, None

    try:
        oldest_visible = min(int(x[0]) for x in initial_arr)
    except Exception:
        return None, None

    if oldest_visible < first_block:
        return None, None

    cursor = oldest_visible - 1
    prev_oldest = oldest_visible
    hops = 0

    while cursor >= first_block and hops < max_hops:
        try:
            hist = await st_archive.get_revealed_commitment_by_hotkey(
                netuid=netuid,
                hotkey_ss58_address=hotkey,
                block=cursor,
            )
        except Exception as e:
            logger.debug(
                "[Registry] archive backfill error hk=%s block=%s: %s",
                hotkey,
                cursor,
                e,
            )
            return None, None

        hist = list(hist or [])
        if not hist:
            return None, None

        blk, _data, obj = _pick_latest_miner_commit_for_element(hist, wanted_element_id)
        if obj is not None:
            return int(blk or 0), obj

        if len(hist) < 10:
            return None, None

        try:
            oldest_hist = min(int(x[0]) for x in hist)
        except Exception:
            return None, None
        if oldest_hist < first_block:
            return None, None
        if oldest_hist >= prev_oldest:
            return None, None

        prev_oldest = oldest_hist
        cursor = oldest_hist - 1
        hops += 1

    return None, None


def _build_miner_candidate(uid: int, hotkey: str, obj: dict, block: int) -> Miner | None:
    model = obj.get("model")
    revision = obj.get("revision")
    slug = obj.get("slug")
    chute_id = obj.get("chute_id")
    committed_eid = obj.get("element_id")
    committed_eid = str(committed_eid).strip() if committed_eid is not None else None
    if not slug:
        return None
    return Miner(
        uid=uid,
        hotkey=hotkey,
        model=model,
        revision=revision,
        slug=slug,
        chute_id=chute_id,
        block=int(block or 0),
        element_id=committed_eid,
    )


def _join_key_to_base(index_url: str, key_or_url: str) -> str:
    key_or_url = str(key_or_url or "").strip()
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        return key_or_url
    base = index_url.rsplit("/", 1)[0] + "/"
    if key_or_url.startswith("/"):
        u = urlparse(index_url)
        return f"{u.scheme}://{u.netloc}{key_or_url}"
    return urljoin(base, key_or_url)


async def _hotkeys_with_prior_scores_for_element(
    *,
    netuid: int,
    element_id: str,
) -> set[str] | None:
    """
    Return hotkeys that already appear in validator index entries for this element.
    Returns None on lookup failure (caller should fail-open).
    """
    try:
        validator_indexes = await get_validator_indexes_from_chain(netuid)
    except Exception as e:
        logger.debug("[Registry] unable to read validator indexes from chain: %s", e)
        return None
    if not validator_indexes:
        return None

    safe_elem = str(element_id or "").strip().replace("/", "_")
    if not safe_elem:
        return None
    elem_seg = f"/manako/{safe_elem}/"
    found: set[str] = set()
    timeout = aiohttp.ClientTimeout(total=_REGISTRY_BACKFILL_INDEX_TIMEOUT_S)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _validator_hk, index_url in validator_indexes.items():
                try:
                    async with session.get(index_url) as resp:
                        if resp.status != 200:
                            continue
                        idx = await resp.json()
                except Exception:
                    continue

                keys: list[str] = []
                if isinstance(idx, list):
                    keys = [_join_key_to_base(index_url, k) for k in idx if isinstance(k, str)]
                elif isinstance(idx, dict) and isinstance(idx.get("entries"), list):
                    for entry in idx.get("entries", []):
                        p = entry.get("path")
                        if isinstance(p, str):
                            keys.append(_join_key_to_base(index_url, p))

                for u in keys:
                    try:
                        path = urlparse(u).path
                    except Exception:
                        continue
                    if elem_seg not in path:
                        continue
                    parts = [p for p in path.split("/") if p]
                    try:
                        root_i = parts.index("manako")
                    except ValueError:
                        continue
                    if len(parts) <= root_i + 3:
                        continue
                    hk = parts[root_i + 3]
                    if hk:
                        found.add(hk)
    except Exception as e:
        logger.debug("[Registry] element score index lookup failed: %s", e)
        return None
    return found

# ---------------------------- Miner registry main ----------------------------- #
async def get_miners_from_registry(
    netuid: int,
    *,
    element_id: str | None = None,
    first_block: int | None = None,
    max_model_size_mb: float | None = None,
    onnx_only: bool | None = None,
    blacklisted_hotkeys: set[str] | None = None,
) -> tuple[Dict[int, Miner], Dict[int, Miner]]:
    """
    Reads on-chain commitments, verifies HF gating/revision, optional HF repo size
    cap, optional ONNX-only model artifact policy, and Chutes slug; then returns at most one miner per model
    (earliest block wins).
    """
    settings = get_settings()
    mechid = settings.SCOREVISION_MECHID

    if blacklisted_hotkeys is None:
        blacklisted_hotkeys = set()
    if blacklisted_hotkeys:
        logger.info("[Registry] loaded %d blacklisted hotkeys", len(blacklisted_hotkeys))

    try:
        st = await get_subtensor()
    except Exception as e:
        logger.warning(
            "[Registry] failed to initialize subtensor (netuid=%s mechid=%s): %s",
            netuid,
            mechid,
            e,
        )
        reset_subtensor()
        return {}, {}

    logger.info(
        "[Registry] extracting candidates (netuid=%s mechid=%s element_id=%s)",
        netuid,
        mechid,
        element_id,
    )

    try:
        meta = await st.metagraph(netuid, mechid=mechid)
        commits = await st.get_all_revealed_commitments(netuid)
    except Exception as e:
        logger.warning("[Registry] error while fetching metagraph/commitments: %s", e)
        reset_subtensor()
        return {}, {}

    # 1) Extract candidates (uid -> Miner)
    candidates: Dict[int, Miner] = {}
    wanted = str(element_id).strip() if element_id is not None else None
    resolved_first_block = (
        int(first_block)
        if first_block is not None
        else _REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK
    )
    unresolved_for_backfill: list[tuple[int, str, list]] = []
    for uid, hk in enumerate(meta.hotkeys):
        bypass_registry_checks = is_registry_bypass(uid, hk)
        if hk in blacklisted_hotkeys and not bypass_registry_checks:
            logger.debug("[Registry] skipping blacklisted hotkey=%s", hk)
            continue
        if hk in blacklisted_hotkeys and bypass_registry_checks:
            logger.info("[Registry] uid=%s hotkey=%s bypassed blacklist", uid, hk)
        arr = commits.get(hk)
        if not arr:
            continue
        best_blk, _best_data, obj = _pick_latest_miner_commit_for_element(arr, wanted)
        if obj is None and wanted is not None:
            unresolved_for_backfill.append((uid, hk, list(arr)))
            continue
        if obj is None:
            continue

        cand = _build_miner_candidate(uid, hk, obj, int(best_blk or 0))
        if cand is not None:
            candidates[uid] = cand

    if (
        wanted is not None
        and _REGISTRY_COMMIT_BACKFILL_ENABLE
        and unresolved_for_backfill
        and _REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT
    ):
        scored_hotkeys = await _hotkeys_with_prior_scores_for_element(
            netuid=netuid,
            element_id=wanted,
        )
        if scored_hotkeys is not None:
            before = len(unresolved_for_backfill)
            unresolved_for_backfill = [
                (uid_i, hk_i, arr_i)
                for (uid_i, hk_i, arr_i) in unresolved_for_backfill
                if hk_i in scored_hotkeys
            ]
            logger.info(
                "[Registry] backfill prefilter by score index element=%s kept=%d/%d unresolved hotkeys",
                wanted,
                len(unresolved_for_backfill),
                before,
            )
        if not unresolved_for_backfill:
            logger.info("[Registry] no unresolved hotkeys left after score-index prefilter")
        else:
            logger.info(
                "[Registry] archive backfill enabled for element=%s unresolved_hotkeys=%d endpoint=%s",
                wanted,
                len(unresolved_for_backfill),
                _REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT,
            )
            st_archive = None
            try:
                st_archive = async_subtensor(_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT)
                await asyncio.wait_for(st_archive.initialize(), timeout=20.0)
                sem = asyncio.Semaphore(_REGISTRY_COMMIT_BACKFILL_CONCURRENCY)

                async def _resolve_one(uid_hk_arr: tuple[int, str, list]):
                    uid_i, hk_i, arr_i = uid_hk_arr
                    async with sem:
                        blk_i, obj_i = await _find_miner_commit_via_archive_backfill(
                            st_archive,
                            netuid=netuid,
                            hotkey=hk_i,
                            initial_arr=arr_i,
                            wanted_element_id=wanted,
                            max_hops=_REGISTRY_COMMIT_BACKFILL_MAX_HOPS,
                            first_block=resolved_first_block,
                        )
                    return uid_i, hk_i, blk_i, obj_i

                results = await asyncio.gather(
                    *[_resolve_one(item) for item in unresolved_for_backfill],
                    return_exceptions=True,
                )

                added = 0
                for item in results:
                    if isinstance(item, Exception):
                        logger.debug("[Registry] archive backfill worker failed: %s", item)
                        continue
                    uid_i, hk_i, blk_i, obj_i = item
                    if obj_i is None or blk_i is None:
                        continue

                    cand = _build_miner_candidate(uid_i, hk_i, obj_i, int(blk_i))
                    if cand is not None:
                        candidates[uid_i] = cand
                        added += 1
                logger.info("[Registry] archive backfill added %d candidate(s)", added)
            except Exception as e:
                logger.warning("[Registry] archive backfill disabled due to error: %s", e)
            finally:
                if st_archive is not None and hasattr(st_archive, "close"):
                    try:
                        await st_archive.close()
                    except Exception:
                        pass

    logger.info("[Registry] %d on-chain candidates", len(candidates))
    if not candidates:
        logger.warning("[Registry] No on-chain candidates")
        return {}, {}

    def _mark_skipped(uid: int, miner: Miner, reason: str) -> None:
        miner.registry_skip_reason = reason
        skipped[uid] = miner

    # 2) Filter by HF gating/inaccessible + Chutes slug/revision checks
    filtered: Dict[int, Miner] = {}
    skipped: Dict[int, Miner] = {}
    for uid, m in candidates.items():
        if is_registry_bypass(uid, m.hotkey):
            logger.info("[Registry] uid=%s hotkey=%s bypassed registry filters", uid, m.hotkey)
            filtered[uid] = m
            continue

        gated = await _hf_gated_or_inaccessible(m.model, m.revision)
        if gated is True:
            logger.info("[Registry] uid=%s slug=%s skipped: HF gated/inaccessible", uid, m.slug)
            _mark_skipped(uid, m, "hf_gated_or_revision_inaccessible")
            continue
        if max_model_size_mb is not None and max_model_size_mb > 0:
            if not m.model:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: missing HF model id for size check",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "missing_hf_model_id_for_size_check")
                continue
            model_size_mb = _hf_repo_total_size_mb(m.model, m.revision)
            if model_size_mb is None:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: unable to resolve HF repo size",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "hf_repo_size_unresolved")
                continue
            if model_size_mb > max_model_size_mb:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: HF repo size %.2fMB exceeds max %.2fMB",
                    uid,
                    m.slug,
                    model_size_mb,
                    max_model_size_mb,
                )
                _mark_skipped(
                    uid,
                    m,
                    f"hf_repo_size_exceeds_max:{model_size_mb:.2f}>{max_model_size_mb:.2f}",
                )
                continue

        if onnx_only is True:
            if not m.model:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: missing HF model id for onnx check",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "missing_hf_model_id_for_onnx_check")
                continue
            model_is_onnx_only = _hf_repo_has_only_onnx_models(m.model, m.revision)
            if model_is_onnx_only is not True:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: HF repo is not onnx-only",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "hf_repo_not_onnx_only")
                continue

        ok = True
        chute_reason = None
        if m.chute_id:
            try:
                info = await fetch_chute_info(m.chute_id)
            except Exception as e:
                logger.info("[Registry] uid=%s slug=%s: Chutes lookup error: %s", uid, m.slug, e)
                info = None
            if not info:
                logger.info("[Registry] uid=%s slug=%s: Chutes unfetched", uid, m.slug)
                ok = False
                chute_reason = "chutes_unfetched"
            else:
                slug_chutes = (info.get("slug") or "").strip()
                if slug_chutes and slug_chutes != (m.slug or ""):
                    ok = False
                    chute_reason = f"chutes_slug_mismatch:{slug_chutes}!={m.slug or ''}"
                    logger.info(
                        "[Registry] uid=%s: slug mismatch (chutes=%s, commit=%s)",
                        uid,
                        slug_chutes,
                        m.slug,
                    )
                ch_rev = info.get("revision")
                if ch_rev and m.revision and str(ch_rev) != str(m.revision):
                    ok = False
                    chute_reason = f"chutes_revision_mismatch:{ch_rev}!={m.revision}"
                    logger.info(
                        "[Registry] uid=%s: revision mismatch (chutes=%s, commit=%s)",
                        uid,
                        ch_rev,
                        m.revision,
                    )

        if ok:
            filtered[uid] = m
        else:
            _mark_skipped(uid, m, chute_reason or "chutes_validation_failed")

    logger.info("[Registry] %d miners after filtering", len(filtered))
    if not filtered:
        logger.warning("[Registry] Filter produced no eligible miners")
        return {}, skipped

    # 3) De-duplicate by model: keep earliest block per model (stable)
    best_by_model: Dict[str, Tuple[int, int]] = {}
    for uid, m in filtered.items():
        dedup_key = m.model
        if not dedup_key and is_registry_bypass(uid, m.hotkey):
            dedup_key = f"__bypass_uid_{uid}"
        if not dedup_key:
            continue
        blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
        prev = best_by_model.get(dedup_key)
        if prev is None or blk < prev[0]:
            best_by_model[dedup_key] = (blk, uid)

    keep_uids = {uid for _, uid in best_by_model.values()}
    kept = {uid: filtered[uid] for uid in keep_uids if uid in filtered}
    dedup_skipped = {}
    for uid, miner in filtered.items():
        if uid in keep_uids:
            continue
        winner = best_by_model.get(miner.model or "")
        if winner is not None:
            winner_blk, winner_uid = winner
            miner.registry_skip_reason = (
                f"dedup_by_model_kept_uid:{winner_uid}_block:{winner_blk}"
            )
        else:
            miner.registry_skip_reason = "dedup_by_model"
        dedup_skipped[uid] = miner
    skipped.update(dedup_skipped)
    logger.info("[Registry] %d miners kept after de-dup by model", len(kept))
    logger.info("[Registry] %d miners skipped", len(skipped))

    return kept, skipped
