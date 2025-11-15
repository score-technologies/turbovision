from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bittensor import wallet as bt_wallet

from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    _wait_n_blocks,
)
from scorevision.utils.huggingface_helpers import get_huggingface_repo_name

logger = logging.getLogger(__name__)

async def get_miner_hotkey_ss58() -> str:
    """
    Small helper to expose the current miner hotkey (ss58) for CLI tools.
    """
    _w, hk = await _get_wallet_and_hotkey()
    return hk

# ---------------------------------------------------------------------------
# Dataclass for local proofs
# ---------------------------------------------------------------------------


@dataclass
class ElementCommitmentProof:
    """
    Local record of a miner's commitment to one or more elements
    for a specific window.
    """

    hotkey: str
    window_id: str
    element_ids: List[str]
    model: str
    revision: Optional[str]
    chute_slug: Optional[str]
    chute_id: Optional[str]
    service_cap: Optional[int]
    block: Optional[int]
    ts: float
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Local persistence helpers
# ---------------------------------------------------------------------------


def _commitments_dir() -> Path:
    settings = get_settings()
    root = settings.SCOREVISION_LOCAL_ROOT
    d = root / "commitments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _commitments_path_for_hotkey(hotkey: str) -> Path:
    safe_hk = hotkey.replace("/", "_")
    return _commitments_dir() / f"{safe_hk}.json"


def _load_local_proofs(hotkey: str) -> List[ElementCommitmentProof]:
    p = _commitments_path_for_hotkey(hotkey)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
        out: List[ElementCommitmentProof] = []
        for item in raw:
            try:
                out.append(
                    ElementCommitmentProof(
                        hotkey=item.get("hotkey", ""),
                        window_id=item.get("window_id", ""),
                        element_ids=list(item.get("element_ids") or []),
                        model=item.get("model", ""),
                        revision=item.get("revision"),
                        chute_slug=item.get("chute_slug"),
                        chute_id=item.get("chute_id"),
                        service_cap=item.get("service_cap"),
                        block=item.get("block"),
                        ts=item.get("ts", 0.0),
                        payload=item.get("payload") or {},
                    )
                )
            except Exception as e:
                logger.debug(
                    "[commitments] failed to load one proof from %s: %s", p, e
                )
        return out
    except Exception as e:
        logger.warning("[commitments] failed to read %s: %s", p, e)
        return []


def _save_local_proofs(hotkey: str, proofs: List[ElementCommitmentProof]) -> None:
    p = _commitments_path_for_hotkey(hotkey)
    try:
        data = [asdict(pr) for pr in proofs]
        p.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
    except Exception as e:
        logger.warning("[commitments] failed to write %s: %s", p, e)


def list_local_commitments(hotkey: str) -> List[ElementCommitmentProof]:
    """
    Public helper to inspect locally stored proofs.
    """
    return _load_local_proofs(hotkey)


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------


async def _get_wallet_and_hotkey() -> tuple[Any, str]:
    """
    Load the miner wallet and return (wallet, hotkey_ss58).
    """
    settings = get_settings()
    w = bt_wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    return w, w.hotkey.ss58_address


def _build_commitment_payload(
    *,
    hotkey_ss58: str,
    window_id: str,
    element_ids: Iterable[str],
    model: str,
    revision: Optional[str],
    chute_id: Optional[str],
    chute_slug: Optional[str],
    service_cap: Optional[int],
    action: str = "commit",
) -> Dict[str, Any]:
    """
    Build the JSON payload that will be sent on-chain via set_reveal_commitment.

    `action` can be "commit" or "withdraw" (soft withdraw semantics).
    """
    ids = [str(e) for e in element_ids]
    payload: Dict[str, Any] = {
        "role": "miner",
        "hotkey": hotkey_ss58,
        "window_id": window_id,
        "element_ids": ids,
        "model": model,
        "revision": revision,
        "chute_id": chute_id,
        "slug": chute_slug,
        "service_cap": service_cap,
        "action": action,
        "version": 1,
    }
    return payload


async def get_commitments_for_hotkey_from_chain(
    hotkey_ss58: str,
    *,
    window_id: Optional[str] = None,
) -> List[ElementCommitmentProof]:
    """
    Inspect on-chain revealed commitments for this hotkey.

    This is read-only and does not depend on local cache.
    """
    settings = get_settings()
    st = await get_subtensor()
    commits = await st.get_all_revealed_commitments(settings.SCOREVISION_NETUID)

    arr = commits.get(hotkey_ss58)
    if not arr:
        return []

    proofs: List[ElementCommitmentProof] = []
    for tup in arr:
        try:
            blk, data = tup
        except Exception:
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue
        if obj.get("role") != "miner":
            continue

        wid = obj.get("window_id")
        if window_id is not None and wid != window_id:
            continue

        el_ids = obj.get("element_ids") or []
        if not isinstance(el_ids, list):
            el_ids = [el_ids]

        proofs.append(
            ElementCommitmentProof(
                hotkey=hotkey_ss58,
                window_id=wid or "",
                element_ids=[str(e) for e in el_ids],
                model=obj.get("model") or "",
                revision=obj.get("revision"),
                chute_slug=obj.get("slug"),
                chute_id=obj.get("chute_id"),
                service_cap=obj.get("service_cap"),
                block=int(blk) if isinstance(blk, int) else None,
                ts=time.time(),
                payload=obj,
            )
        )
    return proofs


# ---------------------------------------------------------------------------
# Public API: commit & withdraw
# ---------------------------------------------------------------------------


async def commit_to_elements(
    element_ids: Iterable[str],
    *,
    window_id: str,
    revision: Optional[str],
    chute_slug: Optional[str],
    chute_id: Optional[str],
    service_cap: Optional[int] = None,
    skip_onchain: bool = False,
    blocks_until_reveal: int = 1,
    wait_blocks_for_confirm: int = 2,
) -> ElementCommitmentProof:
    """
    Commit this miner to the given element_ids for a specific window.

    - Builds a JSON payload and submits it via set_reveal_commitment.
    - Optionally waits for a few blocks so that the commitment is visible.
    - Persists a local proof under SCOREVISION_LOCAL_ROOT/commitments/<hotkey>.json
    """
    settings = get_settings()
    w, hk = await _get_wallet_and_hotkey()
    model = get_huggingface_repo_name()
    payload = _build_commitment_payload(
        hotkey_ss58=hk,
        window_id=window_id,
        element_ids=element_ids,
        model=model,
        revision=revision,
        chute_id=chute_id,
        chute_slug=chute_slug,
        service_cap=service_cap,
        action="commit",
    )

    logger.info(
        "[commitments] committing to elements for window %s: hotkey=%s, elements=%s, model=%s, revision=%s, chute_slug=%s chute_id=%s service_cap=%s",
        window_id,
        hk,
        payload.get("element_ids"),
        model,
        revision,
        chute_slug,
        chute_id,
        service_cap,
    )
    logger.debug("[commitments] payload=%s", payload)

    block_seen: Optional[int] = None

    if not skip_onchain:
        try:
            st = await get_subtensor()
            await st.set_reveal_commitment(
                wallet=w,
                netuid=settings.SCOREVISION_NETUID,
                data=json.dumps(payload, separators=(",", ":")),
                blocks_until_reveal=max(1, int(blocks_until_reveal)),
            )
            logger.info("[commitments] set_reveal_commitment submitted.")

            # Wait a couple of blocks so the reveal is visible for audit.
            await _wait_n_blocks(max(1, int(wait_blocks_for_confirm)))

            # Try to find the commitment we just submitted on-chain.
            chain_proofs = await get_commitments_for_hotkey_from_chain(
                hk, window_id=window_id
            )
            if chain_proofs:
                # Take the last one for that window_id
                block_seen = chain_proofs[-1].block
        except Exception as e:
            logger.warning(
                "[commitments] on-chain commit failed (but local proof will be recorded): %s: %s",
                type(e).__name__,
                e,
            )
    else:
        logger.warning(
            "[commitments] skip_onchain=True → not calling set_reveal_commitment. Recording dry-run proof only."
        )

    proof = ElementCommitmentProof(
        hotkey=hk,
        window_id=window_id,
        element_ids=[str(e) for e in element_ids],
        model=model,
        revision=revision,
        chute_slug=chute_slug,
        chute_id=chute_id,
        service_cap=service_cap,
        block=block_seen,
        ts=time.time(),
        payload=payload,
    )

    # Append to local proofs file
    proofs = _load_local_proofs(hk)
    proofs.append(proof)
    _save_local_proofs(hk, proofs)

    return proof


async def withdraw_commitment(
    element_ids: Iterable[str],
    *,
    window_id: str,
    revision: Optional[str],
    chute_slug: Optional[str],
    chute_id: Optional[str],
    service_cap: Optional[int] = None,
    reason: Optional[str] = None,
    skip_onchain: bool = False,
    blocks_until_reveal: int = 1,
    wait_blocks_for_confirm: int = 2,
) -> ElementCommitmentProof:
    """
    Soft-withdraw commitment to the given element_ids for a specific window.

    Implemented as another set_reveal_commitment call with action="withdraw".
    Validators can interpret this as "do not consider this miner for this
    element/window".
    """
    settings = get_settings()
    w, hk = await _get_wallet_and_hotkey()
    model = get_huggingface_repo_name()
    payload = _build_commitment_payload(
        hotkey_ss58=hk,
        window_id=window_id,
        element_ids=element_ids,
        model=model,
        revision=revision,
        chute_id=chute_id,
        chute_slug=chute_slug,
        service_cap=service_cap,
        action="withdraw",
    )
    if reason:
        payload["reason"] = str(reason)

    logger.info(
        "[commitments] withdrawing commitment for window %s: hotkey=%s, elements=%s, reason=%s",
        window_id,
        hk,
        payload.get("element_ids"),
        reason,
    )
    logger.debug("[commitments] withdraw payload=%s", payload)

    block_seen: Optional[int] = None
    if not skip_onchain:
        try:
            st = await get_subtensor()
            await st.set_reveal_commitment(
                wallet=w,
                netuid=settings.SCOREVISION_NETUID,
                data=json.dumps(payload, separators=(",", ":")),
                blocks_until_reveal=max(1, int(blocks_until_reveal)),
            )
            logger.info("[commitments] withdraw set_reveal_commitment submitted.")
            await _wait_n_blocks(max(1, int(wait_blocks_for_confirm)))

            chain_proofs = await get_commitments_for_hotkey_from_chain(
                hk, window_id=window_id
            )
            if chain_proofs:
                block_seen = chain_proofs[-1].block
        except Exception as e:
            logger.warning(
                "[commitments] on-chain withdraw failed (but local proof will be recorded): %s: %s",
                type(e).__name__,
                e,
            )
    else:
        logger.warning(
            "[commitments] skip_onchain=True → not calling set_reveal_commitment for withdraw."
        )

    proof = ElementCommitmentProof(
        hotkey=hk,
        window_id=window_id,
        element_ids=[str(e) for e in element_ids],
        model=model,
        revision=revision,
        chute_slug=chute_slug,
        chute_id=chute_id,
        service_cap=service_cap,
        block=block_seen,
        ts=time.time(),
        payload=payload,
    )

    proofs = _load_local_proofs(hk)
    proofs.append(proof)
    _save_local_proofs(hk, proofs)

    return proof

async def get_active_element_ids_by_hotkey(
    window_id: str,
) -> Dict[str, Dict[str, ElementCommitmentProof]]:

    settings = get_settings()
    st = await get_subtensor()
    commits = await st.get_all_revealed_commitments(settings.SCOREVISION_NETUID)

    latest_by_key: Dict[tuple[str, str], tuple[int, dict]] = {}

    for hk, arr in commits.items():
        if not arr:
            continue
        for tup in arr:
            try:
                blk, data = tup
            except Exception:
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue
            if obj.get("role") != "miner":
                continue

            wid = obj.get("window_id")
            if wid != window_id:
                continue

            el_ids = obj.get("element_ids") or []
            if not isinstance(el_ids, list):
                el_ids = [el_ids]

            try:
                blk_int = int(blk) if isinstance(blk, int) else 0
            except Exception:
                blk_int = 0

            for e in el_ids:
                eid = str(e)
                key = (hk, eid)
                prev = latest_by_key.get(key)
                if prev is None or blk_int >= prev[0]:
                    latest_by_key[key] = (blk_int, obj)

    result: Dict[str, Dict[str, ElementCommitmentProof]] = {}
    now_ts = time.time()

    for (hk, eid), (blk, obj) in latest_by_key.items():
        action = (obj.get("action") or "commit").lower()
        if action == "withdraw":
            continue

        el_ids = obj.get("element_ids") or []
        if not isinstance(el_ids, list):
            el_ids = [el_ids]

        proof = ElementCommitmentProof(
            hotkey=hk,
            window_id=window_id,
            element_ids=[str(e) for e in el_ids],
            model=obj.get("model") or "",
            revision=obj.get("revision"),
            chute_slug=obj.get("slug"),
            chute_id=obj.get("chute_id"),
            service_cap=obj.get("service_cap"),
            block=blk,
            ts=now_ts,
            payload=obj,
        )
        result.setdefault(hk, {})[eid] = proof

    return result