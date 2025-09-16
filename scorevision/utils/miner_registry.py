from __future__ import annotations
import os, json, time, asyncio, requests
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import aiohttp
from huggingface_hub import HfApi

from scorevision.utils.bittensor_helpers import get_subtensor
from scorevision.utils.settings import get_settings


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
            return bool(r.json().get("gated", False))
    except Exception:
        pass
    return None


def _hf_revision_accessible(model_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return True
    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
        return True
    except Exception:
        return False


def _hf_gated_or_inaccessible(
    model_id: Optional[str], revision: Optional[str]
) -> Optional[bool]:
    if not model_id:
        return True  # no model id -> treat as not eligible
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    gated = None
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
    else:
        gated = _hf_is_gated(model_id)
        # store something even if None to avoid hammering
        _HF_MODEL_GATING_CACHE[model_id] = (
            bool(gated) if gated is not None else False,
            now,
        )
    if gated is True:
        return True
    if not _hf_revision_accessible(model_id, revision):
        return True
    return False  # either False or None (unknown) -> allow


# ------------------------------ Chutes helpers -------------------------------- #
async def _chutes_get_json(url: str, headers: Dict[str, str]) -> Optional[dict]:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            try:
                return await r.json()
            except Exception:
                return None


async def fetch_chute_info(chute_id: str) -> Optional[dict]:
    token = os.getenv("CHUTES_API_KEY", "")
    if not token or not chute_id:
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
    st = await get_subtensor()
    meta = await st.metagraph(netuid)
    commits = await st.get_all_revealed_commitments(netuid)

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
            block=int(block or 0) if uid != 0 else 0,  # mirror special-case for uid 0
        )

    if not candidates:
        return {}

    # 2) Filter by HF gating/inaccessible + Chutes slug/revision checks
    filtered: Dict[int, Miner] = {}
    for uid, m in candidates.items():
        gated = _hf_gated_or_inaccessible(m.model, m.revision)
        if gated is True:
            continue

        ok = True
        if m.chute_id:
            info = await fetch_chute_info(m.chute_id)
            if not info:
                ok = False
            else:
                # cross-check slug (light-normalize)
                slug_chutes = (info.get("slug") or "").strip()
                if slug_chutes and slug_chutes != (m.slug or ""):
                    ok = False
                # optional: if chutes reports a revision, ensure it matches miner's revision
                ch_rev = info.get("revision")
                if ch_rev and m.revision and str(ch_rev) != str(m.revision):
                    ok = False
        if ok:
            filtered[uid] = m

    if not filtered:
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
    return {uid: filtered[uid] for uid in keep_uids if uid in filtered}


if __name__ == "__main__":
    from asyncio import run

    print(run(fetch_chute_info(chute_id="f0b4cebd-c0be-5f31-b7c5-c09302014330")))
