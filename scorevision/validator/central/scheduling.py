import asyncio
import signal
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional
from scorevision.utils.manifest import Manifest, load_manifest_from_public_index
from scorevision.utils.windows import get_current_window_id, get_window_start_block

logger = getLogger(__name__)


def to_pos_int(x: object) -> int | None:
    try:
        if x is None or isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x if x > 0 else None
        if isinstance(x, float):
            return int(x) if x > 0 else None
        v = int(str(x).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def extract_element_tempos(
    manifest: Manifest,
    default_tempo: int,
    track_filter: str | None = None,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    elems = getattr(manifest, "elements", None)

    if isinstance(elems, dict):
        for raw_eid, cfg in elems.items():
            track = cfg.get("track") if isinstance(cfg, dict) else getattr(cfg, "track", None)
            if not _track_matches(track, track_filter):
                continue
            eid = str(raw_eid)
            window_block = None
            if isinstance(cfg, dict):
                window_block = cfg.get("window_block") or cfg.get("tempo")
            else:
                window_block = getattr(cfg, "window_block", None) or getattr(cfg, "tempo", None)
            tempo = to_pos_int(window_block) or default_tempo
            result[eid] = tempo
        return result

    if isinstance(elems, (list, tuple)):
        for elem in elems:
            if isinstance(elem, dict):
                track = elem.get("track")
                eid = elem.get("element_id") or elem.get("id")
                window_block = elem.get("window_block") or elem.get("tempo")
            else:
                track = getattr(elem, "track", None)
                eid = getattr(elem, "element_id", None) or getattr(elem, "id", None)
                window_block = getattr(elem, "window_block", None) or getattr(elem, "tempo", None)
            if not _track_matches(track, track_filter) or not eid:
                continue
            tempo = to_pos_int(window_block) or default_tempo
            result[str(eid)] = tempo
        return result

    return result


def _track_matches(track: str | None, track_filter: str | None) -> bool:
    if track_filter == "private":
        return track == "private"
    if track_filter == "open-source":
        return track != "private"
    return True


def cancel_removed_element_tasks(
    element_state: Dict[str, Dict[str, Any]],
    element_tempos: Dict[str, int],
    log_prefix: str = "",
) -> None:
    removed = set(element_state.keys()) - set(element_tempos.keys())
    for element_id in removed:
        entry = element_state.pop(element_id, None)
        if entry and entry.get("task") is not None:
            task = entry["task"]
            if not task.done():
                logger.info("%sCancelling task for removed element_id=%s", log_prefix, element_id)
                task.cancel()


def update_element_state(
    element_state: Dict[str, Dict[str, Any]],
    element_tempos: Dict[str, int],
    block: int,
    log_prefix: str = "",
) -> None:
    for element_id, tempo in element_tempos.items():
        entry = element_state.get(element_id)
        window_id = get_current_window_id(block, tempo=tempo)
        anchor = get_window_start_block(window_id, tempo=tempo)

        if entry is None:
            element_state[element_id] = {"tempo": tempo, "anchor": anchor, "task": None}
            logger.info("%sRegistered element_id=%s tempo=%s anchor=%s", log_prefix, element_id, tempo, anchor)
        else:
            entry["tempo"] = tempo
            entry["anchor"] = anchor


async def load_manifest(path_manifest: Path | None, settings, block: int) -> Manifest:
    if path_manifest is not None:
        return Manifest.load_yaml(path_manifest)
    if getattr(settings, "URL_MANIFEST", None):
        cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
        return await load_manifest_from_public_index(settings.URL_MANIFEST, block_number=block, cache_dir=cache_dir)
    raise RuntimeError("URL_MANIFEST is required when --manifest-path is not provided.")


def setup_shutdown_handler(shutdown_event: asyncio.Event) -> None:
    def handler():
        logger.warning("Received shutdown signal, stopping runner...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handler)
