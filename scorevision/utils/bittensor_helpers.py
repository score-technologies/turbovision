import asyncio
import os
from pathlib import Path
from base64 import b64decode
from json import load, dumps, loads
from logging import getLogger
from traceback import print_exc
from typing import Optional

from substrateinterface import Keypair
from bittensor import wallet, async_subtensor

from scorevision.utils.settings import get_settings
from scorevision.utils.huggingface_helpers import get_huggingface_repo_name

logger = getLogger(__name__)

_SUBTENSOR = None


def _coerce_last_update_value(value) -> Optional[int]:
    """Convert last_update values from numpy / torch / python scalars into int."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_last_update_for_hotkey(
    meta, hotkey: str, pubkey_hex: str | None = None
) -> Optional[int]:
    """
    Return the validator's last_update height regardless of the underlying container type.
    """
    if meta is None or not hotkey:
        return None

    last_update = getattr(meta, "last_update", None)
    if last_update is None:
        return None

    candidate_keys: list[str] = []
    if hotkey:
        candidate_keys.append(hotkey)
    if pubkey_hex:
        variants = {
            pubkey_hex,
            pubkey_hex.lower(),
            pubkey_hex.upper(),
        }
        candidate_keys.extend([v for v in variants if v])
        candidate_keys.extend([f"0x{v}" for v in variants if v])
    seen: set[str] = set()
    if hasattr(last_update, "get"):
        for key in candidate_keys:
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            try:
                value = last_update.get(key)
            except Exception:
                value = None
            coerced = _coerce_last_update_value(value)
            if coerced is not None:
                return coerced

    index: Optional[int] = None
    hotkeys = getattr(meta, "hotkeys", None)
    hotkeys_list: Optional[list[str]] = None
    if hotkeys is not None:
        try:
            hotkeys_list = list(hotkeys)
        except TypeError:
            hotkeys_list = None

    if hotkeys_list:
        try:
            index = hotkeys_list.index(hotkey)
        except ValueError:
            index = None

    if index is None and hotkeys_list:
        for idx, hk in enumerate(hotkeys_list):
            if hk == hotkey:
                index = idx
                break

    if index is None:
        return None

    try:
        value = last_update[index]
    except Exception:
        return None
    return _coerce_last_update_value(value)


def load_hotkey_keypair(wallet_name: str, hotkey_name: str) -> Keypair:
    settings = get_settings()

    wallet_dir = Path(settings.BITTENSOR_WALLET_PATH).expanduser()
    wallet_dir_str = str(wallet_dir).strip()
    if not wallet_dir_str or wallet_dir_str == ".":
        wallet_dir = Path.home() / ".bittensor" / "wallets"

    wallet_name = wallet_name or settings.BITTENSOR_WALLET_COLD
    hotkey_name = hotkey_name or settings.BITTENSOR_WALLET_HOT

    hotkey_dir = wallet_dir / wallet_name / "hotkeys"
    file_path = hotkey_dir / hotkey_name
    try:
        with open(file_path, "r") as file:
            keypair_data = load(file)
        seed = keypair_data["secretSeed"]
        keypair = Keypair.create_from_seed(seed)
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except Exception as e:
        raise ValueError(f"Failed to load keypair: {e} (path={file_path})")


async def get_subtensor():
    global _SUBTENSOR
    settings = get_settings()

    endpoint = settings.BITTENSOR_SUBTENSOR_ENDPOINT
    init_timeout = float(os.getenv("SUBTENSOR_INIT_TIMEOUT_S", "15.0"))

    async def _init(ep: str):
        st = async_subtensor(ep)
        await asyncio.wait_for(st.initialize(), timeout=init_timeout)
        return st

    if _SUBTENSOR is None:
        try:
            logger.info("Initializing subtensor on %s", endpoint)
            _SUBTENSOR = await _init(endpoint)
        except Exception as e:
            logger.error("Subtensor init failed for %s: %s", endpoint, e)
            raise
    return _SUBTENSOR


def reset_subtensor():
    """Clear cached subtensor client so next access reinitializes connection."""
    global _SUBTENSOR
    _SUBTENSOR = None


async def on_chain_commit(
    skip: bool, revision: str, chute_id: str, chute_slug: str | None
) -> None:
    settings = get_settings()
    repo_name = get_huggingface_repo_name()
    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    payload = {
        "model": repo_name,
        "revision": revision,
        "chute_id": chute_id,
        "slug": chute_slug,
        "hotkey": w.hotkey.ss58_address,
    }
    logger.info(f"Commit payload: {payload}")
    try:
        if skip:
            raise Exception(
                f"No chute_id/slug; skipping on-chain commit for now. Payload would be: {payload}"
            )

        sub = await get_subtensor()

        await sub.set_reveal_commitment(
            wallet=w,
            netuid=settings.SCOREVISION_NETUID,
            data=dumps(payload),
            blocks_until_reveal=1,
        )
        logger.info("On-chain commitment submitted.")
    except Exception as e:
        logger.error(f"(Dry-run) On-chain commit skipped: {type(e).__name__}: {e}")


async def _set_weights_with_confirmation(
    wallet,
    netuid: int,
    mechid: int | None,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "[sv-local]",
) -> bool:
    import bittensor as bt

    settings = get_settings()
    confirm_blocks = max(1, int(os.getenv("SIGNER_CONFIRM_BLOCKS", "3")))

    for attempt in range(retries):
        try:
            st = await get_subtensor()
            ref = await st.get_current_block()
            # soumission (sync) via client non-async
            success, message = bt.subtensor(
                os.getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney")
            ).set_weights(
                wallet=wallet,
                netuid=netuid,
                mechid=mechid if mechid is not None else settings.SCOREVISION_MECHID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=wait_for_inclusion,
            )
            if not success:
                logger.warning(
                    f"{log_prefix} extrinsic submit failed: {message or 'unknown error'}"
                )
            else:
                logger.info(
                    f"{log_prefix} extrinsic submitted; monitoring up to {confirm_blocks} block(s) … (ref {ref}, msg={message or ''})"
                )
                latest_lu = None
                target_mechid = (
                    mechid if mechid is not None else settings.SCOREVISION_MECHID
                )
                hotkey = wallet.hotkey.ss58_address
                for wait_idx in range(confirm_blocks):
                    await st.wait_for_block()
                    meta = await st.metagraph(netuid, mechid=target_mechid)
                    try:
                        meta_hotkeys = getattr(meta, "hotkeys", []) or []
                        try:
                            hotkey_present = hotkey in meta_hotkeys
                        except TypeError:
                            try:
                                hotkey_present = hotkey in list(meta_hotkeys)
                            except TypeError:
                                hotkey_present = False
                        if not hotkey_present:
                            logger.warning(
                                f"{log_prefix} wallet hotkey not found in metagraph; retry…"
                            )
                            break

                        latest_lu = get_last_update_for_hotkey(
                            meta, hotkey, pubkey_hex=wallet.hotkey.public_key.hex()
                        )
                        if latest_lu is None:
                            logger.warning(
                                f"{log_prefix} wallet hotkey found but no last_update entry; retry…"
                            )
                            break
                        if latest_lu >= ref:
                            logger.info(
                                f"{log_prefix} confirmation OK (last_update {latest_lu} >= ref {ref} after {wait_idx + 1} block(s))"
                            )
                            return True
                        logger.debug(
                            f"{log_prefix} waiting for inclusion… (last_update {latest_lu} < ref {ref}, waited {wait_idx + 1}/{confirm_blocks} block(s))"
                        )
                    finally:
                        # Clean up metagraph object to prevent memory accumulation
                        del meta
                if latest_lu is not None:
                    logger.warning(
                        f"{log_prefix} not included after {confirm_blocks} block(s) (last_update {latest_lu} < ref {ref}), retry…"
                    )
                else:
                    logger.warning(
                        f"{log_prefix} not included after {confirm_blocks} block(s) (hotkey missing), retry…"
                    )
        except Exception as e:
            logger.warning(
                f"{log_prefix} attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}"
            )
        await asyncio.sleep(delay_s)
    return False


# --- Validator registry (on-chain) -------------------------------------------


async def on_chain_commit_validator(index_url: str) -> None:
    """ """
    settings = get_settings()
    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    payload = {
        "role": "validator",
        "hotkey": w.hotkey.ss58_address,
        "index_url": index_url,
        "chute_name": settings.CHUTES_USERNAME,
        "version": 1,
    }
    logger.info(f"[validator-commit] {payload}")
    try:
        sub = await get_subtensor()
        await sub.set_reveal_commitment(
            wallet=w,
            netuid=settings.SCOREVISION_NETUID,
            data=dumps(payload),
            blocks_until_reveal=1,
        )
        logger.info("[validator-commit] On-chain commitment submitted.")
    except Exception as e:
        logger.error(f"[validator-commit] failed: {type(e).__name__}: {e}")


async def get_validator_indexes_from_chain(netuid: int | None = None) -> dict[str, str]:
    """ """
    settings = get_settings()
    netuid = netuid if netuid is not None else settings.SCOREVISION_NETUID
    st = await get_subtensor()
    meta = await st.metagraph(netuid, mechid=settings.SCOREVISION_MECHID)
    commits = await st.get_all_revealed_commitments(netuid)

    stake_tensor = getattr(meta, "S", None)
    if stake_tensor is None:
        stake_tensor = getattr(meta, "stake", None)

    result: dict[str, str] = {}
    for i, hk in enumerate(meta.hotkeys):
        arr = commits.get(hk)
        if not arr:
            continue
        _, data = arr[-1]
        try:
            obj = loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("role") == "validator":
            stake_val = 0.0
            try:
                val = stake_tensor[i]
                if hasattr(val, "item"):
                    val = val.item()
                stake_val = float(val)
            except Exception:
                pass

            if stake_val >= 1000.0:
                url = obj.get("index_url")
                if isinstance(url, str) and url.startswith("http"):
                    result[hk] = url

    return result


async def _already_committed_same_index(netuid: int, index_url: str) -> bool:
    """ """
    settings = get_settings()
    st = await get_subtensor()
    meta = await st.metagraph(netuid, mechid=settings.SCOREVISION_MECHID)
    commits = await st.get_all_revealed_commitments(netuid)

    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    hk = w.hotkey.ss58_address
    arr = commits.get(hk)
    if not arr:
        return False
    try:
        _, data = arr[-1]
        obj = loads(data)
    except Exception:
        return False
    return (
        isinstance(obj, dict)
        and obj.get("role") == "validator"
        and str(obj.get("index_url")) == str(index_url)
    )


async def _first_commit_block_by_miner(
    netuid: int,
    *,
    retries: int = 2,
) -> dict[str, int]:
    """"""
    attempt = 0
    while True:
        try:
            st = await get_subtensor()
            settings = get_settings()

            meta = await st.metagraph(netuid, mechid=settings.SCOREVISION_MECHID)
            commits = await st.get_all_revealed_commitments(netuid)

            last_block_by_hk: dict[str, int] = {}
            for hk in meta.hotkeys:
                arr = commits.get(hk)
                if not arr:
                    continue

                last_block = None
                for tup in arr:
                    try:
                        blk, data = tup
                    except Exception:
                        continue

                    try:
                        obj = loads(data)
                    except Exception:
                        continue

                    if isinstance(obj, dict) and obj.get("role") == "validator":
                        continue

                    try:
                        blk_int = int(blk)
                    except Exception:
                        continue

                    if last_block is None or blk_int > last_block:
                        last_block = blk_int

                if last_block is not None:
                    last_block_by_hk[hk] = last_block

            return last_block_by_hk

        except Exception as e:
            attempt += 1
            logger.warning(
                "[first_commit_block_by_miner] error on attempt %d/%d: %s: %s — resetting subtensor",
                attempt,
                retries,
                type(e).__name__,
                e,
            )
            reset_subtensor()

            if attempt > retries:
                raise

            await asyncio.sleep(1.0)


async def _wait_n_blocks(n: int, timeout_per_block: float = 30.0) -> None:
    """Wait for n new blocks on the current subtensor client."""
    if n <= 0:
        return
    st = await get_subtensor()
    for i in range(n):
        try:
            await asyncio.wait_for(st.wait_for_block(), timeout=timeout_per_block)
        except asyncio.TimeoutError:
            logger.warning(
                "[commit-retry] wait_for_block timed out (i=%d/%d), continuing…",
                i + 1,
                n,
            )
            continue


async def on_chain_commit_validator_retry(
    index_url: str,
    *,
    wait_blocks: int = 100,
    confirm_after: int = 3,
    max_retries: int | None = None,
) -> bool:
    """ """
    settings = get_settings()
    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    if await _already_committed_same_index(settings.SCOREVISION_NETUID, index_url):
        logger.info("[validator-commit] Already published %s; skipping.", index_url)
        return True

    attempt = 0
    while True:
        attempt += 1
        try:
            sub = await get_subtensor()
            logger.info("[validator-commit] attempt #%d submitting…", attempt)
            await sub.set_reveal_commitment(
                wallet=w,
                netuid=settings.SCOREVISION_NETUID,
                data=dumps(
                    {
                        "role": "validator",
                        "hotkey": w.hotkey.ss58_address,
                        "index_url": index_url,
                        "chute_name": settings.CHUTES_USERNAME,
                        "version": 1,
                    }
                ),
                blocks_until_reveal=1,
            )
            logger.info(
                "[validator-commit] submitted; waiting %d block(s) for confirm check…",
                confirm_after,
            )
            await _wait_n_blocks(confirm_after)

            if await _already_committed_same_index(
                settings.SCOREVISION_NETUID, index_url
            ):
                logger.info("[validator-commit] confirmed on-chain.")
                return True

            logger.warning(
                "[validator-commit] not visible yet after %d blocks; will retry after %d more blocks.",
                confirm_after,
                wait_blocks,
            )

        except Exception as e:
            logger.warning(
                "[validator-commit] attempt #%d failed: %s: %s",
                attempt,
                type(e).__name__,
                e,
            )

        if max_retries is not None and attempt >= max_retries:
            logger.error("[validator-commit] giving up after %d attempts.", attempt)
            return False

        await _wait_n_blocks(wait_blocks)
