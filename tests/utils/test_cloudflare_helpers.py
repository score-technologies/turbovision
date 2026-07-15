import asyncio
from scorevision.utils.cloudflare_helpers import (
    _cache_remote_json_array,
    _extract_element_miner_commit_tuple_from_key_or_url,
    _exception_summary,
    _inactive_miners_key,
    _lane_index_key,
    _select_lane_specific_index_url,
    emit_shard,
    put_inactive_miners,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
import scorevision.utils.cloudflare_helpers as cloudflare_helpers


def test_extract_element_miner_commit_tuple_with_commit_folder():
    key = "manako/PlayerDetect_v1@1.0/5Fminer/007471716/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(key) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        7471716,
    )


def test_extract_element_miner_commit_tuple_legacy_layout():
    key = "manako/PlayerDetect_v1@1.0/5Fminer/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(key) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        -1,
    )


def test_extract_element_miner_commit_tuple_from_absolute_url():
    url = "https://pub.r2.dev/manako/PlayerDetect_v1@1.0/5Fminer/007471716/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(url) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        7471716,
    )


def test_select_lane_specific_index_url_public_unchanged():
    assert (
        _select_lane_specific_index_url(
            "https://pub.r2.dev/manako/index.json",
            lane="public",
        )
        == "https://pub.r2.dev/manako/index.json"
    )


def test_select_lane_specific_index_url_private_uses_private_index():
    assert (
        _select_lane_specific_index_url(
            "https://pub.r2.dev/manako/index.json",
            lane="private",
        )
        == "https://pub.r2.dev/manako/indexprivate.json"
    )


def test_lane_index_key_public():
    assert _lane_index_key("public") == "manako/index.json"


def test_lane_index_key_private():
    assert _lane_index_key("private") == "manako/indexprivate.json"


def test_inactive_miners_key_is_next_to_public_index():
    assert _inactive_miners_key() == "manako/inactive_miners.json"


@pytest.mark.asyncio
async def test_put_inactive_miners_merges_with_existing_list(monkeypatch):
    put_object = AsyncMock()
    existing_body = AsyncMock()
    existing_body.read.return_value = (
        b'[{"hotkey":"hk-old","element_id":"element-old","commit_block":100}]'
    )
    get_object = AsyncMock(return_value={"Body": existing_body})

    class FakeClientContext:
        async def __aenter__(self):
            return SimpleNamespace(get_object=get_object, put_object=put_object)

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        cloudflare_helpers,
        "get_settings",
        lambda: SimpleNamespace(SCOREVISION_BUCKET="central-bucket"),
    )
    monkeypatch.setattr(
        cloudflare_helpers,
        "get_s3_client",
        lambda: FakeClientContext(),
    )
    inactive_miners = [
        {"hotkey": "hk1", "element_id": "element-a", "commit_block": 123}
    ]

    key = await put_inactive_miners(inactive_miners)

    assert key == "manako/inactive_miners.json"
    put_object.assert_awaited_once_with(
        Bucket="central-bucket",
        Key="manako/inactive_miners.json",
        Body=(
            '[{"hotkey":"hk-old","element_id":"element-old","commit_block":100},'
            '{"hotkey":"hk1","element_id":"element-a","commit_block":123}]'
        ),
        ContentType="application/json",
    )


def test_exception_summary_includes_type_for_empty_message():
    assert _exception_summary(TimeoutError()) == "TimeoutError"


@pytest.mark.asyncio
async def test_cache_remote_json_array_uses_stale_cache_when_head_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cloudflare_helpers, "_get_cache_dir", lambda: tmp_path)

    url = "https://example.com/manako/evaluation/000000001-a.json"
    cached_path = cloudflare_helpers._cache_path_for_url(url)
    cached_path.write_bytes(b'{"payload":{}}\n')

    async def _fail_head(_url):
        raise TimeoutError()

    async def _fail_get(_url):
        raise AssertionError("GET should not be called when stale cache is available")

    monkeypatch.setattr(cloudflare_helpers, "_http_head_meta", _fail_head)
    monkeypatch.setattr(cloudflare_helpers, "_http_get_json", _fail_get)

    result = await _cache_remote_json_array(url, asyncio.Semaphore(1))

    assert result == cached_path


@pytest.mark.asyncio
async def test_emit_shard_uses_trigger_block_without_subtensor(monkeypatch):
    async def _fail_get_subtensor():
        raise AssertionError("get_subtensor should not be called when trigger_block is set")

    sink_mock = AsyncMock(return_value=("5Fhotkey", [{"signature": "0xdeadbeef"}]))
    monkeypatch.setattr(cloudflare_helpers, "get_subtensor", _fail_get_subtensor)
    monkeypatch.setattr(cloudflare_helpers, "sink_sv_at", sink_mock)
    monkeypatch.setattr(
        cloudflare_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            SCOREVISION_VERSION="test",
            SCOREVISION_PGT_RECIPE_HASH="sha256:test",
        ),
    )

    challenge = SimpleNamespace(
        challenge_id="challenge-1",
        api_task_id="api-1",
        prompt="prompt",
        payload=None,
        env="test",
        meta={},
    )
    miner_run = SimpleNamespace(
        predictions=None,
        success=True,
        latency_ms=10.0,
        latency_p50_ms=8.0,
        latency_p95_ms=12.0,
        latency_p99_ms=15.0,
        latency_max_ms=20.0,
        error=None,
        model="m",
        revision="r",
    )
    evaluation = SimpleNamespace(
        acc_breakdown={},
        acc=1.0,
        score=0.8,
        latency_p95_ms=12.0,
        latency_pass=True,
        rtf=0.1,
        scored_frame_numbers=[],
    )

    await emit_shard(
        slug="slug",
        challenge=challenge,
        miner_run=miner_run,
        evaluation=evaluation,
        miner_hotkey_ss58="5FminerHotkey",
        trigger_block=123,
        element_id="PlayerDetect_v1@1.0",
    )

    sink_mock.assert_awaited_once()
    eval_key = sink_mock.await_args.args[0]
    assert eval_key.endswith("/evaluation/000000123-challenge-1.json")


@pytest.mark.asyncio
async def test_emit_shard_falls_back_to_cached_block_when_subtensor_down(monkeypatch):
    async def _fail_get_subtensor():
        raise TimeoutError("subtensor down")

    sink_mock = AsyncMock(return_value=("5Fhotkey", [{"signature": "0xdeadbeef"}]))
    monkeypatch.setattr(cloudflare_helpers, "get_subtensor", _fail_get_subtensor)
    monkeypatch.setattr(cloudflare_helpers, "sink_sv_at", sink_mock)
    monkeypatch.setattr(
        cloudflare_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            SCOREVISION_VERSION="test",
            SCOREVISION_PGT_RECIPE_HASH="sha256:test",
        ),
    )
    monkeypatch.setattr(cloudflare_helpers, "_last_known_emit_block", 777)

    challenge = SimpleNamespace(
        challenge_id="challenge-2",
        api_task_id="api-2",
        prompt="prompt",
        payload=None,
        env="test",
        meta={},
    )
    miner_run = SimpleNamespace(
        predictions=None,
        success=False,
        latency_ms=0.0,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        latency_max_ms=0.0,
        error="x",
        model="m",
        revision="r",
    )
    evaluation = SimpleNamespace(
        acc_breakdown={},
        acc=0.0,
        score=0.0,
        latency_p95_ms=0.0,
        latency_pass=False,
        rtf=None,
        scored_frame_numbers=[],
    )

    await emit_shard(
        slug="slug",
        challenge=challenge,
        miner_run=miner_run,
        evaluation=evaluation,
        miner_hotkey_ss58="5FminerHotkey",
        element_id="PlayerDetect_v1@1.0",
    )

    sink_mock.assert_awaited_once()
    eval_key = sink_mock.await_args.args[0]
    assert eval_key.endswith("/evaluation/000000777-challenge-2.json")
