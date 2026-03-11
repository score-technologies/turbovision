import logging
import httpx

logger = logging.getLogger(__name__)


async def report_failed_spotcheck(
    api_url: str,
    hotkey: str,
    reason: str,
    auth_token: str,
) -> None:
    if not api_url:
        logger.info("No blacklist API configured, skipping report for %s", hotkey)
        return

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{api_url}/api/blacklist",
                headers=headers,
                json={"hotkey": hotkey, "reason": reason},
            )
            if response.status_code == 200:
                logger.info("Blacklisted %s: %s", hotkey, reason)
            else:
                logger.warning("Blacklist request returned %d", response.status_code)
    except Exception as e:
        logger.error("Failed to blacklist %s: %s", hotkey, e)
