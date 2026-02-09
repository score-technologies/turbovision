from __future__ import annotations

from logging import getLogger
from pathlib import Path

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

    if hotkeys:
        sample = list(hotkeys)[:5]
    return hotkeys
