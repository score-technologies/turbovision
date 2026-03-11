import logging
import httpx

logger = logging.getLogger(__name__)


async def fetch_ground_truth(
    api_url: str,
    challenge_id: str,
    auth_token: str,
) -> list[dict] | None:
    if not api_url:
        return None

    url = f"{api_url}/api/private-track/ground-truth/{challenge_id}"
    headers = {"Authorization": f"Bearer {auth_token}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 404:
                logger.warning("No ground truth found for %s", challenge_id)
                return None
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.error("Failed to fetch ground truth for %s: %s", challenge_id, e)
        return None

    return data.get("ground_truth", [])
