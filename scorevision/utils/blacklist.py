from __future__ import annotations
import json
from logging import getLogger
import httpx
from scorevision.utils.signing import build_validator_query_params

logger = getLogger(__name__)


HARDCODED_BLACKLIST_HOTKEYS: set[str] = {
    "5DvY7cxtAvUeA2Goq26LNyzqSfPyjfY9SUsD4bgJa5PMnVNa",
    "5CMaFwgm2rPka66iUcgAa2SpBPskk6KqAGWZeKVx8APLnqTZ",
    "5CfGbGvZz6YUUPT84ntoGHANy1ddk9xGJiaQZVEb9Qi57Foc",
    "5CG3NLAVRzs1uxqFHTjUaSESW4N7PLFYrRosjuh7JRKFoYmi",
    "5CqKodaU2F5atWnLBivA2TeQWoJ9jxfPwADQ2HAWfAQHNsZV",
    "5CmCgAKAW1B3YmBDeVJxxd2MbhMeC6n1esJsVFdb2qUvD55r",
    "5Fnx48i6A6KtN9a8pAxQDZDQo3Z49JeviaLkcJUbLB9yCM4Q",
    "5HBK3g2PURYZ99ufLRCfoqEga6ScT8KzVerxk54BELF7hZvu",
    "5FZiTikQum61QyzWaaGmsiKmH1eX2jLcZ61V2E2WBg1QAAYV",
    "5C81gi3bXLbAWccH6VFY5T7BNhv9usEMdtwxHUVqeFPJ3zy9",
    "5Gj9pWjksQXkuaoVxHRaKN1pmgQYiUddrNweY3SFBWMGo2QD",
    "5FkPaJTKgBr3rXX1YUt952ejVnwui4RZHoafBoR6txGkrSc3",
    "5GYVPC5tTnHEtqZ4qVCHfyRFmQ1SDHHo9nwECphftrN8wbRG",
    "5FkF684znSsYy1bjsaTDYpZDYN983Z5kavsvv6bNYyuNZQHy",
    "5GNvBDx7qBcUUM3EwLTD2a5qVyFcjKs7Nf2zk2qxL1xPsXCw",
    "5DvuNyrQQuDmzBMUDVNsRJSXBMxuBKAuKiJQUiTMAgTJoDoF",
}


class BlacklistAPI:
    def __init__(self, base_url: str, keypair):
        self.base_url = base_url
        self.keypair = keypair

    async def get_blacklist(self) -> set[str]:
        params = build_validator_query_params(self.keypair)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/api/blacklist",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return {
                    entry.get("hotkey")
                    for entry in data
                    if isinstance(entry, dict) and entry.get("hotkey")
                }
            return set()

    async def add_to_blacklist(self, hotkey: str, reason: str) -> bool:
        body = json.dumps({"hotkey": hotkey, "reason": reason}).encode()
        params = build_validator_query_params(self.keypair)
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/blacklist",
                params=params,
                content=body,
                headers=headers,
            )
            return response.status_code == 200

    async def remove_from_blacklist(self, hotkey: str) -> bool:
        params = build_validator_query_params(self.keypair)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.base_url}/api/blacklist/{hotkey}",
                params=params,
            )
            return response.status_code == 200


async def fetch_blacklisted_hotkeys(blacklist_api: BlacklistAPI | None) -> set[str]:
    if blacklist_api is None:
        return set(HARDCODED_BLACKLIST_HOTKEYS)
    try:
        api_blacklist = await blacklist_api.get_blacklist()
        return set(api_blacklist) | HARDCODED_BLACKLIST_HOTKEYS
    except Exception as e:
        logger.warning("[Blacklist] Failed to fetch blacklist: %s", e)
        return set(HARDCODED_BLACKLIST_HOTKEYS)
