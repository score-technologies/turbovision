from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from scorevision.validator.central.open_source.runner import (
    _cleanup_video_cache,
    _extract_element_id_from_chal_api,
    _enough_bboxes_per_frame,
)


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
