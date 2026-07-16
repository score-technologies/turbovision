#!/usr/bin/env python3
"""Read-only live-chain smoke test for validator deployments."""

import asyncio
import importlib.metadata as metadata
import time

import bittensor as bt

from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.bittensor_commitments import get_all_revealed_commitments
from scorevision.utils.miner_registry import get_miners_from_registry
from scorevision.utils.settings import get_settings
from scorevision.validator.central.private_track.registry import get_registered_miners


async def _timed(label: str, awaitable, timeout: float):
    started = time.monotonic()
    result = await asyncio.wait_for(awaitable, timeout=timeout)
    print(f"[live-read] {label}: OK ({time.monotonic() - started:.2f}s)")
    return result


async def main() -> None:
    settings = get_settings()
    version = metadata.version("bittensor")
    if version != "10.3.0":
        raise RuntimeError(f"expected bittensor 10.3.0, got {version}")

    validator_hotkey = None
    try:
        wallet = bt.Wallet(
            name=settings.BITTENSOR_WALLET_COLD,
            hotkey=settings.BITTENSOR_WALLET_HOT,
        )
        validator_hotkey = wallet.hotkey.ss58_address
    except Exception as error:
        print(f"[live-read] validator wallet unavailable: {type(error).__name__}")
    print(
        "[live-read] config: "
        f"bittensor={version} netuid={settings.SCOREVISION_NETUID} "
        f"mechid={settings.SCOREVISION_MECHID} endpoint={settings.BITTENSOR_SUBTENSOR_ENDPOINT}"
    )
    if validator_hotkey:
        print(f"[live-read] validator hotkey: {validator_hotkey}")

    subtensor = None
    try:
        subtensor = await _timed("subtensor initialize", get_subtensor(), 30.0)
        block = await _timed("get_current_block", subtensor.get_current_block(), 30.0)
        if int(block) <= 0:
            raise RuntimeError(f"invalid current block: {block}")
        print(f"[live-read] current block: {block}")

        metagraph = await _timed(
            "metagraph",
            subtensor.metagraph(
                settings.SCOREVISION_NETUID,
                mechid=settings.SCOREVISION_MECHID,
            ),
            90.0,
        )
        hotkeys = list(metagraph.hotkeys)
        if not hotkeys:
            raise RuntimeError("metagraph contains no hotkeys")
        validator_uid = (
            hotkeys.index(validator_hotkey)
            if validator_hotkey is not None and validator_hotkey in hotkeys
            else None
        )
        print(
            f"[live-read] metagraph neurons={len(hotkeys)} "
            f"validator_uid={validator_uid}"
        )

        commitments = await _timed(
            "get_all_revealed_commitments",
            get_all_revealed_commitments(subtensor, settings.SCOREVISION_NETUID),
            90.0,
        )
        committed_hotkeys = sum(1 for values in commitments.values() if values)
        print(
            f"[live-read] commitments hotkeys={len(commitments)} "
            f"non_empty={committed_hotkeys}"
        )

        private_miners = await _timed(
            "private miner registry",
            get_registered_miners(
                subtensor,
                metagraph,
                blacklist=set(),
                inactive_miner_tuples=set(),
            ),
            90.0,
        )
        print(
            f"[live-read] private registry miners={len(private_miners)} "
            f"uids={[miner.uid for miner in private_miners[:10]]}"
        )

        public_miners, public_skipped = await _timed(
            "public miner registry and external filters",
            get_miners_from_registry(
                settings.SCOREVISION_NETUID,
                blacklisted_hotkeys=set(),
                compliance_failure_tuples=set(),
                inactive_miner_tuples=set(),
            ),
            240.0,
        )
        print(
            f"[live-read] public registry eligible={len(public_miners)} "
            f"skipped={len(public_skipped)} uids={sorted(public_miners)[:10]}"
        )
        print("[live-read] SUCCESS: all live read-only chain operations passed")
    finally:
        if subtensor is not None:
            try:
                await subtensor.close()
            except Exception:
                pass
        reset_subtensor()


if __name__ == "__main__":
    asyncio.run(main())
