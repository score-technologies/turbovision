from __future__ import annotations
import json
from logging import getLogger
import httpx
from scorevision.utils.request_signing import build_signed_headers

logger = getLogger(__name__)


class BlacklistAPI:
    def __init__(self, base_url: str, keypair):
        self.base_url = base_url
        self.keypair = keypair

    async def get_blacklist(self) -> set[str]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/api/blacklist",
                headers=build_signed_headers(self.keypair),
            )
            response.raise_for_status()
            data = response.json()
            return {entry["hotkey"] for entry in data.get("blacklist", [])}

    async def add_to_blacklist(self, hotkey: str, reason: str) -> bool:
        body = json.dumps({"hotkey": hotkey, "reason": reason}).encode()
        headers = build_signed_headers(self.keypair, body)
        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/blacklist",
                content=body,
                headers=headers,
            )
            return response.status_code == 200

    async def remove_from_blacklist(self, hotkey: str) -> bool:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.base_url}/api/blacklist/{hotkey}",
                headers=build_signed_headers(self.keypair),
            )
            return response.status_code == 200


async def fetch_blacklisted_hotkeys(blacklist_api: BlacklistAPI | None) -> set[str]:
    if blacklist_api is None:
        return set()
    try:
        return await blacklist_api.get_blacklist()
    except Exception as e:
        logger.warning("[Blacklist] Failed to fetch blacklist: %s", e)
        return set()
