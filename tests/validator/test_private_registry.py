import json
from types import SimpleNamespace

import pytest

from scorevision.utils.inactive_miners import InactiveMinerTuple
from scorevision.validator.central.private_track.registry import (
    get_registered_miners,
    get_valid_private_commitments,
)


class _FakeSubtensor:
    def __init__(self, commits_by_hotkey: dict[str, list[tuple[int, str]]]):
        self._commits_by_hotkey = commits_by_hotkey

    async def get_all_revealed_commitments(self, netuid):
        return self._commits_by_hotkey


def _build_metagraph(hotkeys: list[str]):
    axons = [SimpleNamespace(ip="127.0.0.1", port=8000 + i) for i in range(len(hotkeys))]
    return SimpleNamespace(
        hotkeys=hotkeys, axons=axons, block_at_registration=[0] * len(hotkeys)
    )


def _private_commit(
    block: int,
    *,
    element_id: str,
    image_tag: str | None = "v1",
) -> tuple[int, str]:
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
@pytest.mark.parametrize("port", [0, 1])
async def test_reset_axon_is_challenged_but_zero_port_is_skipped(monkeypatch, port):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    metagraph.axons[0] = SimpleNamespace(ip="0.0.0.0", port=port)
    subtensor = _FakeSubtensor({"hk1": [_private_commit(110, element_id="E1")]})

    valid = await get_valid_private_commitments(
        subtensor, metagraph, set(), element_id="E1", inactive_miner_tuples=set()
    )
    miners = await get_registered_miners(
        subtensor, metagraph, set(), element_id="E1", inactive_miner_tuples=set()
    )

    assert valid["hk1"][0] == 110
    assert [miner.hotkey for miner in miners] == (["hk1"] if port else [])


@pytest.mark.asyncio
async def test_private_registry_ignores_commit_from_previous_registration(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["old", "current"])
    metagraph.block_at_registration = [200, 100]
    subtensor = _FakeSubtensor(
        {
            "old": [_private_commit(150, element_id="E1")],
            "current": [_private_commit(150, element_id="E1")],
        }
    )

    miners = await get_registered_miners(
        subtensor, metagraph, set(), element_id="E1", inactive_miner_tuples=set()
    )

    assert [miner.hotkey for miner in miners] == ["current"]


@pytest.mark.asyncio
async def test_private_registry_keeps_commit_without_image_tag(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    subtensor = _FakeSubtensor(
        {"hk1": [_private_commit(100, element_id="E1", image_tag=None)]}
    )

    miners = await get_registered_miners(
        subtensor,
        metagraph,
        set(),
        element_id="E1",
    )

    assert len(miners) == 1
    assert miners[0].hotkey == "hk1"
    assert miners[0].image_repo == "ghcr.io/mine/pt-solution"
    assert miners[0].image_tag == ""


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


@pytest.mark.asyncio
async def test_private_registry_ignores_only_exact_inactive_tuple(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1", "hk2"])
    subtensor = _FakeSubtensor(
        {
            "hk1": [_private_commit(110, element_id="E1")],
            "hk2": [_private_commit(125, element_id="E1")],
        }
    )

    miners = await get_registered_miners(
        subtensor,
        metagraph,
        set(),
        element_id="E1",
        inactive_miner_tuples={InactiveMinerTuple("hk1", "E1", 110)},
    )

    assert [(miner.hotkey, miner.commit_block) for miner in miners] == [("hk2", 125)]


@pytest.mark.asyncio
async def test_private_registry_keeps_new_commit_after_inactive_tuple(monkeypatch):
    monkeypatch.setattr(
        "scorevision.validator.central.private_track.registry.get_settings",
        lambda: SimpleNamespace(SCOREVISION_NETUID=18),
    )
    metagraph = _build_metagraph(["hk1"])
    subtensor = _FakeSubtensor(
        {
            "hk1": [
                _private_commit(110, element_id="E1", image_tag="old"),
                _private_commit(120, element_id="E1", image_tag="new"),
            ]
        }
    )

    miners = await get_registered_miners(
        subtensor,
        metagraph,
        set(),
        element_id="E1",
        inactive_miner_tuples={InactiveMinerTuple("hk1", "E1", 110)},
    )

    assert len(miners) == 1
    assert miners[0].commit_block == 120
