from scorevision.miner.private_track.security import BLACKLIST_ENABLED, VERIFY_ENABLED
from scorevision.utils.logging import setup_logging

logger = setup_logging()


def log_startup_config() -> None:
    logger.info(f"Blacklist: {'ENABLED' if BLACKLIST_ENABLED else 'DISABLED'}")
    logger.info(f"Verify: {'ENABLED' if VERIFY_ENABLED else 'DISABLED'}")

    if BLACKLIST_ENABLED or VERIFY_ENABLED:
        logger.info("Security requires: BITTENSOR_WALLET_COLD, HOTKEY, NETUID, MIN_STAKE_THRESHOLD")

    if not BLACKLIST_ENABLED and not VERIFY_ENABLED:
        logger.warning("All security DISABLED - local testing only")
