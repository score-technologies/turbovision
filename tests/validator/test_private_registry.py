import json
from types import SimpleNamespace

import pytest

from scorevision.validator.central.private_track.registry import get_registered_miners


class _FakeSubtensor:
    def __init__(self, commits_by_hotkey: dict[str, list[tuple[int, str]]]):
        self._commits_by_hotkey = commits_by_hotkey

    async def get_all_revealed_commitments(self, netuid):
        return self._commits_by_hotkey


def _build_metagraph(hotkeys: list[str]):
    axons = [SimpleNamespace(ip="127.0.0.1", port=8000 + i) for i in range(len(hotkeys))]
    return SimpleNamespace(hotkeys=hotkeys, axons=axons)


def _private_commit(block: int, *, element_id: str, image_tag: str = "v1") -> tuple[int, str]:
    payload = {
        "role": "miner",
        "track": "private",
        "element_id": element_id,
        "image_repo": "ghcr.io/mine/pt-solution",
        "image_tag": image_tag,
        "hotkey": "ignored-in-test",
    }
    return (block, json.dumps(payload))


@pytest.mark.asyncio
async def test_private_registry_selects_latest_commit_per_requested_element(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    subtensor = _FakeSubtensor(
        {
            "hk1": [
                _private_commit(100, element_id="E1", image_tag="e1-v1"),
                _private_commit(120, element_id="E2", image_tag="e2-v1"),
            ]
        }
    )

    e1_miners = await get_registered_miners(subtensor, metagraph, set(), element_id="E1")
    e2_miners = await get_registered_miners(subtensor, metagraph, set(), element_id="E2")

    assert len(e1_miners) == 1
    assert e1_miners[0].commit_block == 100
    assert e1_miners[0].image_tag == "e1-v1"

    assert len(e2_miners) == 1
    assert e2_miners[0].commit_block == 120
    assert e2_miners[0].image_tag == "e2-v1"


@pytest.mark.asyncio
async def test_private_registry_same_element_uses_latest_block(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    subtensor = _FakeSubtensor(
        {
            "hk1": [
                _private_commit(100, element_id="E1", image_tag="v1"),
                _private_commit(130, element_id="E1", image_tag="v2"),
            ]
        }
    )

    miners = await get_registered_miners(subtensor, metagraph, set(), element_id="E1")

    assert len(miners) == 1
    assert miners[0].commit_block == 130
    assert miners[0].image_tag == "v2"


@pytest.mark.asyncio
async def test_private_registry_two_hotkeys_can_commit_same_element(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1", "hk2"])
    subtensor = _FakeSubtensor(
        {
            "hk1": [_private_commit(110, element_id="E1", image_tag="hk1-v1")],
            "hk2": [_private_commit(125, element_id="E1", image_tag="hk2-v1")],
        }
    )

    miners = await get_registered_miners(subtensor, metagraph, set(), element_id="E1")
    by_hotkey = {m.hotkey: m for m in miners}

    assert set(by_hotkey.keys()) == {"hk1", "hk2"}
    assert by_hotkey["hk1"].commit_block == 110
    assert by_hotkey["hk2"].commit_block == 125


@pytest.mark.asyncio
async def test_private_registry_returns_empty_for_uncommitted_element(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    subtensor = _FakeSubtensor(
        {"hk1": [_private_commit(100, element_id="E1", image_tag="v1")]}
    )

    miners = await get_registered_miners(subtensor, metagraph, set(), element_id="E2")

    assert miners == []
