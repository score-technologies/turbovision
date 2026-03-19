import json
from types import SimpleNamespace

import pytest

from scorevision.utils import miner_registry as registry


class _FakeHfApi:
    def __init__(self, token=None, nodes=None):
        self.token = token
        self._nodes = nodes or []

    def list_repo_tree(self, **kwargs):
        return self._nodes


class _FakeSubtensor:
    async def metagraph(self, netuid, mechid=None):
        return SimpleNamespace(hotkeys=["hk1"])

    async def get_all_revealed_commitments(self, netuid):
        return {
            "hk1": [
                (
                    10,
                    json.dumps(
                        {
                            "role": "miner",
                            "model": "org/model",
                            "revision": "rev1",
                            "slug": "slug1",
                            "chute_id": "chute1",
                            "element_id": "PlayerDetect_v1@1.0",
                        }
                    ),
                )
            ]
        }


@pytest.fixture(autouse=True)
def clear_registry_caches():
    registry._HF_ONNX_ONLY_CACHE.clear()


def test_hf_repo_has_only_onnx_models_true(monkeypatch):
    nodes = [
        SimpleNamespace(path="model.onnx", size=123),
        SimpleNamespace(path="README.md", size=10),
    ]

    monkeypatch.setattr(registry, "HfApi", lambda token=None: _FakeHfApi(token=token, nodes=nodes))

    assert registry._hf_repo_has_only_onnx_models("org/model", "rev1") is True


def test_hf_repo_has_only_onnx_models_false_when_mixed(monkeypatch):
    nodes = [
        SimpleNamespace(path="model.onnx", size=123),
        SimpleNamespace(path="model.safetensors", size=456),
    ]

    monkeypatch.setattr(registry, "HfApi", lambda token=None: _FakeHfApi(token=token, nodes=nodes))

    assert registry._hf_repo_has_only_onnx_models("org/model", "rev1") is False


def test_hf_repo_has_only_onnx_models_false_when_no_model_artifact(monkeypatch):
    nodes = [
        SimpleNamespace(path="README.md", size=10),
        SimpleNamespace(path="config.json", size=20),
    ]

    monkeypatch.setattr(registry, "HfApi", lambda token=None: _FakeHfApi(token=token, nodes=nodes))

    assert registry._hf_repo_has_only_onnx_models("org/model", "rev1") is False


@pytest.mark.asyncio
async def test_get_miners_from_registry_skips_when_onnx_only_enabled_and_repo_not_onnx(monkeypatch):
    async def fake_get_subtensor():
        return _FakeSubtensor()

    async def fake_gated(_model, _revision):
        return False

    async def fake_chute_info(_chute_id):
        return {"slug": "slug1", "revision": "rev1"}

    monkeypatch.setattr(registry, "get_settings", lambda: SimpleNamespace(SCOREVISION_MECHID=1))
    monkeypatch.setattr(registry, "load_blacklisted_hotkeys", lambda: set())
    monkeypatch.setattr(registry, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(registry, "_hf_gated_or_inaccessible", fake_gated)
    monkeypatch.setattr(registry, "_hf_repo_has_only_onnx_models", lambda _m, _r: False)
    monkeypatch.setattr(registry, "fetch_chute_info", fake_chute_info)

    kept, skipped = await registry.get_miners_from_registry(
        18,
        element_id="PlayerDetect_v1@1.0",
        onnx_only=True,
    )

    assert kept == {}
    assert 0 in skipped
    assert skipped[0].registry_skip_reason == "hf_repo_not_onnx_only"


@pytest.mark.asyncio
async def test_get_miners_from_registry_keeps_when_onnx_only_disabled(monkeypatch):
    async def fake_get_subtensor():
        return _FakeSubtensor()

    async def fake_gated(_model, _revision):
        return False

    async def fake_chute_info(_chute_id):
        return {"slug": "slug1", "revision": "rev1"}

    monkeypatch.setattr(registry, "get_settings", lambda: SimpleNamespace(SCOREVISION_MECHID=1))
    monkeypatch.setattr(registry, "load_blacklisted_hotkeys", lambda: set())
    monkeypatch.setattr(registry, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(registry, "_hf_gated_or_inaccessible", fake_gated)
    monkeypatch.setattr(registry, "_hf_repo_has_only_onnx_models", lambda _m, _r: False)
    monkeypatch.setattr(registry, "fetch_chute_info", fake_chute_info)

    kept, skipped = await registry.get_miners_from_registry(
        18,
        element_id="PlayerDetect_v1@1.0",
        onnx_only=None,
    )

    assert 0 in kept
    assert skipped == {}
