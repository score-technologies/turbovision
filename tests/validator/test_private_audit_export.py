from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scorevision.validator.audit.private_track import export as export_mod


def test_winner_target_uses_latest_commit_for_winner():
    target = export_mod._winner_target(
        {
            "winner_hotkey": "hk1",
            "winner_commit_block": 100,
            "winner_1": {"hotkey": "hk1", "commit_block": 120},
            "top_3_official": [
                {"hotkey": "hk1", "commit_block": 110},
                {"hotkey": "hk2", "commit_block": 999},
            ],
        }
    )
    assert target == ("hk1", 120)


def test_candidate_keys_excludes_ten_latest(monkeypatch):
    monkeypatch.setattr(export_mod.random, "shuffle", lambda rows: None)
    keys = [
        f"manako/manako_DetectCricketDelivery/hk1/000000120/evaluation/{block:09d}-{block}.json"
        for block in range(1, 14)
    ]
    candidates = export_mod._candidate_keys(
        keys, "manako/DetectCricketDelivery", "hk1", 120
    )
    assert candidates == keys[:3]


def test_candidate_keys_requires_more_than_ten_samples():
    keys = [
        f"manako/E/hk/000000001/evaluation/{block:09d}-{block}.json"
        for block in range(10)
    ]
    assert export_mod._candidate_keys(keys, "E", "hk", 1) == []


def test_matching_payload_requires_private_exact_tuple():
    payload = {
        "lane": "private",
        "element_id": "E",
        "telemetry": {
            "miner": {
                "hotkey": "hk",
                "commitment": {"commit_block": 12},
            }
        },
    }
    assert export_mod._matching_payload([{"payload": payload}], "E", "hk", 12) == payload
    assert export_mod._matching_payload([{"payload": payload}], "E", "other", 12) is None


def test_shard_references_accepts_current_schema():
    response_key, task_id = export_mod._shard_references(
        {
            "telemetry": {
                "api_task_id": "12345",
                "run": {"responses_key": "private_responses/example.json"},
            }
        }
    )
    assert response_key == "private_responses/example.json"
    assert task_id == "12345"


def test_due_trigger_block_catches_up_to_latest_interval():
    assert export_mod._due_trigger_block(7300, 100, 7200) == 7300
    assert export_mod._due_trigger_block(7299, 100, 7200) is None
    assert export_mod._due_trigger_block(14550, 100, 7200) == 14500


def test_block_from_key_supports_snapshot_and_evaluation_names():
    assert export_mod._block_from_key("manako/winners/008818800.json") == 8818800
    assert (
        export_mod._block_from_key(
            "manako/E/hk/000000001/evaluation/008795400-64734.json"
        )
        == 8795400
    )
    assert export_mod._block_from_key("https://example.test/manako/audit/008820000.json") == 8820000


def test_cricket_audit_keeps_only_positively_weighted_fields():
    miner_response = {
        "challenge_id": "123",
        "miner_hotkey": "hk-cricket",
        "video_url": "https://example.test/cricket.mp4",
        "frames": [{"frame_id": 1, "url": "https://example.test/frame.jpg"}],
        "predictions": [
            {
                "kph": 111,
                "bounce_x": 7.6,
                "stump_y": -0.7,
                "runs": 1,
                "wickets": 0,
                "inningsid": 1,
            }
        ],
    }
    groundtruth = {
        "challenge_id": 123,
        "ground_truth": [
            {
                "meta": {
                    "kph": 110.9,
                    "bounce_x": 7.7,
                    "stump_y": -0.71,
                    "runs": 2,
                    "wickets": 1,
                    "innings_id": "1",
                },
                "frame": 0,
                "action": "delivery",
            }
        ],
    }

    prediction, expected = export_mod._format_cricket_audit(
        miner_response, groundtruth
    )

    assert prediction == {
        "miner_hotkey": "hk-cricket",
        "video_url": "https://example.test/cricket.mp4",
        "prediction": {"kph": 111, "bounce_x": 7.6, "stump_y": -0.7},
    }
    assert expected == {"kph": 110.9, "bounce_x": 7.7, "stump_y": -0.71}


def test_football_audit_normalizes_both_sides_to_frame_action():
    miner_response = {
        "challenge_id": "456",
        "miner_hotkey": "hk-football",
        "video_url": None,
        "frames": [{"frame_id": 1, "url": "https://example.test/frame.jpg"}],
        "predictions": [
            {"frame": 109, "action": "pass", "confidence": 0.66},
            {"frame": 155, "action": "pass_received", "confidence": 0.84},
        ],
    }
    groundtruth = {
        "challenge_id": 456,
        "ground_truth": [
            {
                "frame": 108,
                "action": "pass",
                "team": "AWAY",
                "player_id": 1,
                "meta": {"x": 0.3},
            },
            {
                "frame": 154,
                "action": "pass_received",
                "team": "AWAY",
                "player_id": 2,
                "meta": {"x": 0.2},
            },
        ],
    }

    predictions, expected = export_mod._format_football_audit(
        miner_response, groundtruth
    )

    assert predictions == {
        "miner_hotkey": "hk-football",
        "frames": [{"frame_id": 1, "url": "https://example.test/frame.jpg"}],
        "predictions": [
            {"frame": 109, "action": "pass"},
            {"frame": 155, "action": "pass_received"},
        ],
    }
    assert expected == [
        {"frame": 108, "action": "pass"},
        {"frame": 154, "action": "pass_received"},
    ]


def test_miner_response_context_omits_null_asset():
    assert export_mod._miner_response_context(
        {"miner_hotkey": "hk", "video_url": None, "frames": None}
    ) == {"miner_hotkey": "hk"}


def test_private_response_config_requires_read_credentials(monkeypatch):
    secret = lambda value: SimpleNamespace(get_secret_value=lambda: value)
    monkeypatch.setattr(
        export_mod,
        "get_settings",
        lambda: SimpleNamespace(
            PRIVATE_RESPONSES_R2_BUCKET="private",
            PRIVATE_RESPONSES_R2_ACCOUNT_ID=secret("account"),
            PRIVATE_RESPONSES_R2_READ_ACCESS_KEY_ID=secret("read-id"),
            PRIVATE_RESPONSES_R2_READ_SECRET_ACCESS_KEY=secret("read-secret"),
            CENTRAL_R2_CONCURRENCY=4,
        ),
    )
    config = export_mod._private_responses_config()
    assert config.access_key_id == "read-id"
    assert config.secret_access_key == "read-secret"


@pytest.mark.asyncio
async def test_private_manifest_uses_dedicated_index(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        PRIVATE_AUDIT_MANIFEST_INDEX_URL="https://private.example/manifest/index.json",
        SCOREVISION_CACHE_DIR=tmp_path,
    )
    manifest = SimpleNamespace(
        elements=[
            SimpleNamespace(
                id="private/E1",
                track="private",
                groundtruth_type=SimpleNamespace(value="soccer_action"),
            ),
            SimpleNamespace(id="public/E2", track="public", groundtruth_type=None),
        ]
    )
    load_manifest = AsyncMock(return_value=manifest)
    monkeypatch.setattr(export_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(export_mod, "load_manifest_from_public_index", load_manifest)

    result = await export_mod._private_manifest_elements(123)

    assert result == {"private/E1": "soccer_action"}
    load_manifest.assert_awaited_once_with(
        "https://private.example/manifest/index.json",
        block_number=123,
        cache_dir=tmp_path,
    )
