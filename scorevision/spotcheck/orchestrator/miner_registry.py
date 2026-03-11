import json
import logging
from dataclasses import dataclass
import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredMiner:
    uid: int
    hotkey: str
    image_repo: str
    image_tag: str


async def fetch_registered_miners(
    netuid: int,
    network: str,
) -> dict[str, RegisteredMiner]:
    subtensor = bt.AsyncSubtensor(network=network)
    metagraph = await subtensor.metagraph(netuid)

    try:
        commitments = await subtensor.get_all_revealed_commitments(netuid)
    except Exception as e:
        logger.error("Failed to fetch commitments: %s", e)
        return {}

    miners: dict[str, RegisteredMiner] = {}

    for uid, hotkey in enumerate(metagraph.hotkeys):
        commitment = commitments.get(hotkey)
        if not commitment:
            continue

        _block, raw_data = commitment[-1]
        parsed = _parse_private_commitment(raw_data)
        if not parsed:
            continue

        image_repo, image_tag = parsed
        miners[hotkey] = RegisteredMiner(
            uid=uid,
            hotkey=hotkey,
            image_repo=image_repo,
            image_tag=image_tag,
        )

    logger.info("Found %d registered private track miners", len(miners))
    return miners


def _parse_private_commitment(raw_data: str) -> tuple[str, str] | None:
    try:
        obj = json.loads(raw_data)
    except (json.JSONDecodeError, TypeError):
        return None

    if obj.get("track") != "private":
        return None

    image_repo = obj.get("image_repo")
    image_tag = obj.get("image_tag")

    if not image_repo or not image_tag:
        return None

    return image_repo, image_tag
