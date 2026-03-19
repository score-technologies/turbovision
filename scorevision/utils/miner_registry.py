from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from logging import getLogger

import aiohttp
from huggingface_hub import HfApi

from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.blacklist import load_blacklisted_hotkeys
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

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


async def fetch_chute_info(chute_id: str) -> Optional[dict]:
    token = os.getenv("CHUTES_API_KEY", "")
    if not token or not chute_id:
        logger.debug("[Chutes] missing token or chute_id")
        return None
    return await _chutes_get_json(
        f"https://api.chutes.ai/chutes/{chute_id}",
        headers={"Authorization": token},
    )

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
        if role and role != "miner":
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

# ---------------------------- Miner registry main ----------------------------- #
async def get_miners_from_registry(
    netuid: int,
    *,
    element_id: str | None = None,
    max_model_size_mb: float | None = None,
    onnx_only: bool | None = None,
) -> tuple[Dict[int, Miner], Dict[int, Miner]]:
    """
    Reads on-chain commitments, verifies HF gating/revision, optional HF repo size
    cap, optional ONNX-only model artifact policy, and Chutes slug; then returns at most one miner per model
    (earliest block wins).
    """
    settings = get_settings()
    mechid = settings.SCOREVISION_MECHID

    blacklisted_hotkeys = load_blacklisted_hotkeys()
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
        wanted = str(element_id).strip() if element_id is not None else None
        best_blk, _best_data, obj = _pick_latest_miner_commit_for_element(arr, wanted)
        if obj is None:
            continue

        model = obj.get("model")
        revision = obj.get("revision")
        slug = obj.get("slug")
        chute_id = obj.get("chute_id")
        committed_eid = obj.get("element_id")
        committed_eid = str(committed_eid).strip() if committed_eid is not None else None

        if not slug:
            continue

        candidates[uid] = Miner(
            uid=uid,
            hotkey=hk,
            model=model,
            revision=revision,
            slug=slug,
            chute_id=chute_id,
            block=int(best_blk or 0),
            element_id=committed_eid,
        )

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
            info = await fetch_chute_info(m.chute_id)
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
