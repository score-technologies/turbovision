from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Optional

WINDOW_BLOCK_PREFIX = "block-"
DATE_WINDOW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_current_window_id(block_number: int, tempo: int = 300) -> str:
    """
    Compute a deterministic window ID from the current block height.

    Canonical block-based format:
      - "block-{start_block}"

    Example:
      block_number=1234, tempo=300 -> "block-1200"
    """
    if block_number < 0:
        raise ValueError("block_number must be non-negative")

    start_block = (block_number // tempo) * tempo
    return f"{WINDOW_BLOCK_PREFIX}{start_block}"


def get_window_start_block(window_id: str, tempo: int = 300) -> Optional[int]:
    """
    Recover the start block from a block-based window_id ("block-{start}").

    Returns None for non block-based formats (e.g. date-based "YYYY-MM-DD").
    """
    if window_id.startswith(WINDOW_BLOCK_PREFIX):
        try:
            return int(window_id[len(WINDOW_BLOCK_PREFIX) :])
        except ValueError:
            return None

    # Date-based format (YYYY-MM-DD) – supported as an ID, but doesn't map
    # directly to a block height here, so we return None.
    if DATE_WINDOW_RE.match(window_id):
        return None

    return None


def is_window_active(
    window_id: str,
    current_block: int,
    expiry_block: Optional[int],
    tempo: int = 300,
) -> bool:
    """
    Return True if the window is considered active at current_block.

    Rules:
      - If expiry_block is set and current_block > expiry_block → inactive.
      - For block-based IDs, we also require current_block >= start_block.
      - For other formats, we only use expiry_block.
    """
    if expiry_block is not None and current_block > expiry_block:
        return False

    start_block = get_window_start_block(window_id, tempo=tempo)
    if start_block is not None and current_block < start_block:
        return False

    return True
