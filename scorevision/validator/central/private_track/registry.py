import json
import logging
from dataclasses import dataclass
import bittensor as bt
from scorevision.utils.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RegisteredMiner:
    uid: int
    hotkey: str
    ip: str
    port: int
    image_repo: str
    image_tag: str
    commit_block: int


async def get_registered_miners(
    subtensor: bt.AsyncSubtensor,
    metagraph,
    blacklist: set[str],
) -> list[RegisteredMiner]:
    settings = get_settings()
    netuid = settings.SCOREVISION_NETUID

    miners = []

    try:
        commits = await subtensor.get_all_revealed_commitments(netuid)
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

        block, data = commitment[-1]
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("track") != "private":
            continue

        image_repo = obj.get("image_repo")
        image_tag = obj.get("image_tag")

        if not image_repo or not image_tag:
            continue

        miners.append(RegisteredMiner(
            uid=uid,
            hotkey=hotkey,
            ip=axon.ip,
            port=int(axon.port),
            image_repo=image_repo,
            image_tag=image_tag,
            commit_block=int(block),
        ))

    logger.info("Found %d registered private track miners", len(miners))
    return miners
