import pytest

from scorevision.utils import inactive_miners as inactive_miners_module
from scorevision.utils.inactive_miners import (
    DEFAULT_INACTIVE_MINERS_URL,
    InactiveMinerTuple,
    fetch_inactive_miner_tuples,
    is_inactive_miner_tuple,
    parse_inactive_miner_tuples,
)


def test_parse_inactive_miner_tuples_accepts_only_valid_rows():
    parsed = parse_inactive_miner_tuples(
        [
            {"hotkey": "hk1", "element_id": "element-a", "commit_block": 10},
            ["hk2", "element-b", 20],
            {"hotkey": "", "element_id": "element-c", "commit_block": 30},
            {"hotkey": "hk3", "element_id": "element-c"},
        ]
    )

    assert parsed == {
        InactiveMinerTuple("hk1", "element-a", 10),
        InactiveMinerTuple("hk2", "element-b", 20),
    }


def test_inactive_match_requires_all_three_values():
    inactive = {InactiveMinerTuple("hk1", "element-a", 10)}

    assert is_inactive_miner_tuple(
        inactive,
        hotkey="hk1",
        element_id="element-a",
        commit_block=10,
    )
    assert not is_inactive_miner_tuple(
        inactive,
        hotkey="hk1",
        element_id="element-a",
        commit_block=11,
    )
    assert not is_inactive_miner_tuple(
        inactive,
        hotkey="hk1",
        element_id="element-b",
        commit_block=10,
    )


def test_default_inactive_miners_url_uses_turbo_prefix():
    assert (
        DEFAULT_INACTIVE_MINERS_URL
        == "https://turbo.scoredata.me/manako/inactive_miners.json"
    )


@pytest.mark.asyncio
async def test_fetch_inactive_miners_uses_fixed_turbo_url(monkeypatch):
    requested_urls = []

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self):
            return [
                {"hotkey": "hk1", "element_id": "element-a", "commit_block": 10}
            ]

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def get(self, url):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(
        inactive_miners_module.aiohttp,
        "ClientSession",
        lambda timeout: FakeSession(),
    )
    inactive_miners_module._FETCH_CACHE.clear()

    result = await fetch_inactive_miner_tuples(use_cache=False)

    assert result == {InactiveMinerTuple("hk1", "element-a", 10)}
    assert requested_urls == [DEFAULT_INACTIVE_MINERS_URL]
