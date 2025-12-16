from __future__ import annotations
import os, json, time, asyncio, requests
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from logging import getLogger

import aiohttp
from huggingface_hub import HfApi

from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass
class Miner:
    uid: int
    hotkey: str
    model: Optional[str]
    revision: Optional[str]
    slug: Optional[str]
    chute_id: Optional[str]
    block: int


# ------------------------- HF gating & revision checks ------------------------- #
_HF_MODEL_GATING_CACHE: Dict[str, Tuple[bool, float]] = {}
_HF_GATING_TTL = 300  # seconds


def _hf_is_gated(model_id: str) -> Optional[bool]:
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=5)
        if r.status_code == 200:
            gated = bool(r.json().get("gated", False))
            logger.debug("[HF] model=%s gated=%s", model_id, gated)
            return gated
        logger.debug("[HF] model=%s status=%s", model_id, r.status_code)
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


def _hf_gated_or_inaccessible(
    model_id: Optional[str], revision: Optional[str]
) -> Optional[bool]:
    if not model_id:
        logger.debug("[HF] no model id → treat as not eligible")
        return True  # no model id -> treat as not eligible
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
        logger.debug("[HF] cache hit model=%s gated=%s", model_id, gated)
    else:
        gated = _hf_is_gated(model_id)
        _HF_MODEL_GATING_CACHE[model_id] = (
            bool(gated) if gated is not None else False,
            now,
        )
        logger.debug("[HF] cache set model=%s gated=%s", model_id, gated)

    if gated is True:
        logger.info("[HF] model=%s is gated", model_id)
        return True
    if not _hf_revision_accessible(model_id, revision):
        logger.info("[HF] model=%s revision inaccessible", model_id)
        return True
    return False  # either False or None (unknown) -> allow


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


# ---------------------------- Miner registry main ----------------------------- #
async def get_miners_from_registry(netuid: int) -> Dict[int, Miner]:
    """
    Reads on-chain commitments, verifies HF gating/revision and Chutes slug,
    and returns at most one miner per model (earliest block wins).
    """
    settings = get_settings()
    mechid = settings.SCOREVISION_MECHID

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
        return {}

    logger.info(
        "[Registry] extracting candidates (netuid=%s mechid=%s)", netuid, mechid
    )

    try:
        meta = await st.metagraph(netuid, mechid=mechid)
        commits = await st.get_all_revealed_commitments(netuid)
    except Exception as e:
        logger.warning(
            "[Registry] error while fetching metagraph/commitments: %s", e
        )
        reset_subtensor()
        return {}

    # 1) Extract candidates (uid -> Miner)
    candidates: Dict[int, Miner] = {}
    for uid, hk in enumerate(meta.hotkeys):
        arr = commits.get(hk)
        if not arr:
            continue
        block, data = arr[-1]
        try:
            obj = json.loads(data)
        except Exception:
            logger.debug("[Registry] uid=%s hotkey=%s invalid JSON", uid, hk)
            continue

        model = obj.get("model")
        revision = obj.get("revision")
        slug = obj.get("slug")
        chute_id = obj.get("chute_id")

        if not slug:
            # no slug -> cannot call this miner
            continue

        candidates[uid] = Miner(
            uid=uid,
            hotkey=hk,
            model=model,
            revision=revision,
            slug=slug,
            chute_id=chute_id,
            block=int(block or 0) if uid != 0 else 0,
        )

    logger.info("[Registry] %d on-chain candidates", len(candidates))
    if not candidates:
        logger.warning("[Registry] No on-chain candidates")
        return {}

    # 2) Filter by HF gating/inaccessible + Chutes slug/revision checks
    filtered: Dict[int, Miner] = {}
    for uid, m in candidates.items():
        gated = _hf_gated_or_inaccessible(m.model, m.revision)
        if gated is True:
            logger.info(
                "[Registry] uid=%s slug=%s skipped: HF gated/inaccessible", uid, m.slug
            )
            continue

        ok = True
        if m.chute_id:
            info = await fetch_chute_info(m.chute_id)
            if not info:
                logger.info("[Registry] uid=%s slug=%s: Chutes unfetched", uid, m.slug)
                ok = False
            else:
                slug_chutes = (info.get("slug") or "").strip()
                if slug_chutes and slug_chutes != (m.slug or ""):
                    ok = False
                    logger.info(
                        "[Registry] uid=%s: slug mismatch (chutes=%s, commit=%s)",
                        uid,
                        slug_chutes,
                        m.slug,
                    )
                ch_rev = info.get("revision")
                if ch_rev and m.revision and str(ch_rev) != str(m.revision):
                    ok = False
                    logger.info(
                        "[Registry] uid=%s: revision mismatch (chutes=%s, commit=%s)",
                        uid,
                        ch_rev,
                        m.revision,
                    )
        if ok:
            filtered[uid] = m

    logger.info("[Registry] %d miners after filtering", len(filtered))
    if not filtered:
        logger.warning("[Registry] Filter produced no eligible miners")
        return {}

    # 3) De-duplicate by model: keep earliest block per model (stable)
    best_by_model: Dict[str, Tuple[int, int]] = {}
    for uid, m in filtered.items():
        if not m.model:
            continue
        blk = (
            m.block
            if isinstance(m.block, int)
            else (int(m.block) if m.block is not None else (2**63 - 1))
        )
        prev = best_by_model.get(m.model)
        if prev is None or blk < prev[0]:
            best_by_model[m.model] = (blk, uid)

    keep_uids = {uid for _, uid in best_by_model.values()}
    kept = {uid: filtered[uid] for uid in keep_uids if uid in filtered}
    logger.info("[Registry] %d miners kept after de-dup by model", len(kept))
    return kept
