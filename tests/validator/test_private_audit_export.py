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
