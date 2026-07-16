import json
import logging
from dataclasses import dataclass
import bittensor as bt
from scorevision.utils.inactive_miners import (
    InactiveMinerTuple,
    fetch_inactive_miner_tuples,
    is_inactive_miner_tuple,
)
from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_commitments import get_all_revealed_commitments

logger = logging.getLogger(__name__)


@dataclass
class RegisteredMiner:
    uid: int
    hotkey: str
    ip: str
    port: int
    image_repo: str
    image_tag: str
    image_digest: str
    commit_block: int


def _pick_latest_private_commit_for_element(
    commitments: list[tuple[int, str]],
    wanted_element_id: str | None,
) -> tuple[int | None, dict | None]:
    wanted = str(wanted_element_id).strip() if wanted_element_id is not None else None
    best_block: int | None = None
    best_obj: dict | None = None

    for block, data in commitments:
        try:
            block_i = int(block)
        except Exception:
            continue

        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("track") != "private":
            continue
        if obj.get("role") not in (None, "miner"):
            continue

        committed_element_id = obj.get("element_id")
        committed_element_id = (
            str(committed_element_id).strip() if committed_element_id is not None else None
        )
        if wanted is not None and committed_element_id != wanted:
            continue

        if best_block is None or block_i > best_block:
            best_block = block_i
            best_obj = obj

    return best_block, best_obj


async def get_registered_miners(
    subtensor: bt.AsyncSubtensor,
    metagraph,
    blacklist: set[str],
    element_id: str | None = None,
    inactive_miner_tuples: set[InactiveMinerTuple] | None = None,
) -> list[RegisteredMiner]:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID

    if inactive_miner_tuples is None and element_id is not None:
        inactive_miner_tuples = await fetch_inactive_miner_tuples()

    miners = []

    try:
        commits = await get_all_revealed_commitments(subtensor, netuid)
    except Exception as e:
        logger.error("Failed to fetch commitments: %s", e)
        return []

    for uid, hotkey in enumerate(metagraph.hotkeys):
        if hotkey in blacklist:
            continue

        axon = metagraph.axons[uid]
        if not axon.ip or not axon.port:
            continue

        commitment = commits.get(hotkey)
        if not commitment:
            continue

        block, obj = _pick_latest_private_commit_for_element(commitment, element_id)
        if obj is None or block is None:
            continue
        if is_inactive_miner_tuple(
            inactive_miner_tuples,
            hotkey=hotkey,
            element_id=element_id,
            commit_block=block,
        ):
            logger.info(
                "Private registry ignored hotkey=%s element_id=%s commit_block=%s: inactive miner tuple",
                hotkey,
                element_id,
                block,
            )
            continue

        image_repo = obj.get("image_repo")
        image_tag = obj.get("image_tag")
        image_digest = obj.get("image_digest", "")

        if not image_repo or not image_tag:
            continue

        miners.append(RegisteredMiner(
            uid=uid,
            hotkey=hotkey,
            ip=axon.ip,
            port=int(axon.port),
            image_repo=image_repo,
            image_tag=image_tag,
            image_digest=image_digest,
            commit_block=int(block),
        ))

    logger.info("Found %d registered private track miners", len(miners))
    return miners
