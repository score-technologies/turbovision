from __future__ import annotations

from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

_DEFAULT_BLACKLIST_PATH = Path(__file__).resolve().parents[2] / "blacklist"


def load_blacklisted_hotkeys(path: Path | str | None = None) -> set[str]:
    blacklist_path = Path(path) if path is not None else _DEFAULT_BLACKLIST_PATH
    try:
        content = blacklist_path.read_text(encoding="utf-8")
    except FileNotFoundError:
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
        logger.debug("[Blacklist] Loaded %d hotkeys from %s", len(hotkeys), blacklist_path)
    return hotkeys
