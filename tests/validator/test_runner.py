import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from scorevision.validator.central.open_source.runner import (
    _cleanup_video_cache,
    _extract_element_id_from_chal_api,
    _extract_element_tempos_from_manifest,
    _to_pos_int,
    _enough_bboxes_per_frame,
)


def test_to_pos_int_with_positive_int():
    assert _to_pos_int(5) == 5
    assert _to_pos_int(1) == 1
    assert _to_pos_int(100) == 100


def test_to_pos_int_with_zero_or_negative():
    assert _to_pos_int(0) is None
    assert _to_pos_int(-1) is None
    assert _to_pos_int(-100) is None


def test_to_pos_int_with_float():
    assert _to_pos_int(5.7) == 5
    assert _to_pos_int(1.0) == 1
    assert _to_pos_int(0.9) is None


def test_to_pos_int_with_string():
    assert _to_pos_int("5") == 5
    assert _to_pos_int("100") == 100
    assert _to_pos_int("0") is None
    assert _to_pos_int("-5") is None
    assert _to_pos_int("abc") is None


def test_to_pos_int_with_none_and_bool():
    assert _to_pos_int(None) is None
    assert _to_pos_int(True) is None
    assert _to_pos_int(False) is None


def test_extract_element_id_from_chal_api_direct():
    chal_api = {"element_id": "soccer_detect"}
    assert _extract_element_id_from_chal_api(chal_api) == "soccer_detect"


def test_extract_element_id_from_chal_api_nested_element():
    chal_api = {"element": {"element_id": "pitch_detect"}}
    assert _extract_element_id_from_chal_api(chal_api) == "pitch_detect"

    chal_api = {"element": {"id": "pitch_detect_v2"}}
    assert _extract_element_id_from_chal_api(chal_api) == "pitch_detect_v2"


def test_extract_element_id_from_chal_api_nested_meta():
    chal_api = {"meta": {"element_id": "from_meta"}}
    assert _extract_element_id_from_chal_api(chal_api) == "from_meta"

    chal_api = {"meta": {"element": "from_meta_element"}}
    assert _extract_element_id_from_chal_api(chal_api) == "from_meta_element"


def test_extract_element_id_from_chal_api_not_found():
    assert _extract_element_id_from_chal_api({}) is None
    assert _extract_element_id_from_chal_api({"other": "data"}) is None
    assert _extract_element_id_from_chal_api(None) is None


def test_extract_element_tempos_from_manifest_dict_elements():
    manifest = SimpleNamespace(
        elements={
            "soccer_detect": {"window_block": 100},
            "pitch_detect": {"tempo": 200},
            "player_track": {},
        }
    )
    result = _extract_element_tempos_from_manifest(manifest, default_tempo_blocks=300)
    assert result == {
        "soccer_detect": 100,
        "pitch_detect": 200,
        "player_track": 300,
    }


def test_extract_element_tempos_from_manifest_list_elements():
    manifest = SimpleNamespace(
        elements=[
            SimpleNamespace(element_id="elem1", window_block=150),
            SimpleNamespace(id="elem2", tempo=250),
            {"element_id": "elem3", "window_block": 350},
            {"id": "elem4"},
        ]
    )
    result = _extract_element_tempos_from_manifest(manifest, default_tempo_blocks=500)
    assert result["elem1"] == 150
    assert result["elem2"] == 250
    assert result["elem3"] == 350
    assert result["elem4"] == 500


def test_extract_element_tempos_from_manifest_no_elements():
    manifest = SimpleNamespace(elements=None)
    result = _extract_element_tempos_from_manifest(manifest, default_tempo_blocks=300)
    assert result == {}


def test_cleanup_video_cache_with_store():
    mock_store = MagicMock()
    video_cache = {"store": mock_store, "other": "data"}
    _cleanup_video_cache(video_cache, None)
    mock_store.unlink.assert_called_once()
    assert video_cache == {}


def test_cleanup_video_cache_with_frame_store():
    mock_frame_store = MagicMock()
    video_cache = {"other": "data"}
    _cleanup_video_cache(video_cache, mock_frame_store)
    mock_frame_store.unlink.assert_called_once()
    assert video_cache == {}


def test_cleanup_video_cache_with_path(tmp_path):
    test_file = tmp_path / "test_video.mp4"
    test_file.touch()
    assert test_file.exists()

    video_cache = {"path": str(test_file)}
    _cleanup_video_cache(video_cache, None)
    assert not test_file.exists()
    assert video_cache == {}


def test_cleanup_video_cache_empty():
    video_cache = {}
    _cleanup_video_cache(video_cache, None)
    assert video_cache == {}


def test_enough_bboxes_per_frame_sufficient():
    annotations = [
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6])),
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6, 7])),
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6, 7, 8])),
    ]
    assert _enough_bboxes_per_frame(
        annotations, min_bboxes_per_frame=6, min_frames_required=3
    )


def test_enough_bboxes_per_frame_insufficient_bboxes():
    annotations = [
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3])),
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6])),
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2])),
    ]
    assert not _enough_bboxes_per_frame(
        annotations, min_bboxes_per_frame=6, min_frames_required=3
    )


def test_enough_bboxes_per_frame_insufficient_frames():
    annotations = [
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6])),
        SimpleNamespace(annotation=SimpleNamespace(bboxes=[1, 2, 3, 4, 5, 6])),
    ]
    assert not _enough_bboxes_per_frame(
        annotations, min_bboxes_per_frame=6, min_frames_required=3
    )


def test_enough_bboxes_per_frame_empty():
    assert not _enough_bboxes_per_frame(
        [], min_bboxes_per_frame=6, min_frames_required=1
    )

