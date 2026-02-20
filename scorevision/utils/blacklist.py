from __future__ import annotations
import time
from logging import getLogger
from pathlib import Path
import httpx

logger = getLogger(__name__)

_DEFAULT_BLACKLIST_PATH = Path("/app/blacklist")


def load_blacklisted_hotkeys(path: Path | str | None = None) -> set[str]:
    blacklist_path = _DEFAULT_BLACKLIST_PATH

    try:
        content = blacklist_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("[Blacklist] File not found: %s", blacklist_path)
        return set()
    except Exception as e:
        logger.warning("[Blacklist] Failed to read %s: %s", blacklist_path, e)
        return set()

    hotkeys: set[str] = set()
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hotkeys.add(s)

    return hotkeys


class BlacklistAPI:
    def __init__(self, base_url: str, hotkey):
        self.base_url = base_url
        self.hotkey = hotkey

    def _auth_headers(self) -> dict:
        nonce = str(int(time.time() * 1e9))
        signature = f"0x{self.hotkey.sign(nonce.encode('utf-8')).hex()}"
        return {
            "X-Validator-Hotkey": self.hotkey.ss58_address,
            "X-Signature": signature,
            "X-Nonce": nonce,
        }

    async def get_blacklist(self) -> set[str]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/api/blacklist",
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return {entry["hotkey"] for entry in data.get("blacklist", [])}

    async def add_to_blacklist(self, hotkey: str, reason: str) -> bool:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/blacklist",
                headers=self._auth_headers(),
                json={"hotkey": hotkey, "reason": reason},
            )
            return response.status_code == 200

    async def remove_from_blacklist(self, hotkey: str) -> bool:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.base_url}/api/blacklist/{hotkey}",
                headers=self._auth_headers(),
            )
            return response.status_code == 200
