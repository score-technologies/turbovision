import logging
import httpx

logger = logging.getLogger(__name__)


async def fetch_pending_spotchecks(api_url: str, auth_token: str) -> list[dict]:
    url = f"{api_url}/api/spotchecks/pending"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.error("Failed to fetch pending spotchecks: %s", e)
        return []

    targets = []
    for item in data:
        targets.append({
            "miner_hotkey": item["miner_hotkey"],
            "image_repo": item["miner_image_repo"],
            "image_tag": item["miner_image_tag"],
            "image_digest": item.get("miner_image_digest", ""),
            "scoring_version": item.get("scoring_version", 0),
            "challenge_id": item["challenge_id"],
            "video_url": item["challenge_url"],
            "original_score": item["original_score"],
        })

    logger.info("Fetched %d pending spotchecks", len(targets))
    return targets
