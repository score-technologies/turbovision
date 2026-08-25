import json
from types import SimpleNamespace

import pytest

from scorevision.utils import bittensor_helpers
from scorevision.utils.commit_recovery import RecoveredCommitment


@pytest.mark.asyncio
async def test_on_chain_commit_recover_submits_minimal_payload(monkeypatch):
    captured = {}

    class FakeWallet:
        def __init__(self, name, hotkey):
            self.name = name
            self.hotkey = SimpleNamespace(ss58_address="hk1")

    class FakeSubtensor:
        async def set_reveal_commitment(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True, message="")

    async def fake_get_subtensor():
        return FakeSubtensor()

    monkeypatch.setattr(bittensor_helpers, "Wallet", FakeWallet)
    monkeypatch.setattr(bittensor_helpers, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(
        bittensor_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            BITTENSOR_WALLET_COLD="cold",
            BITTENSOR_WALLET_HOT="hot",
            SCOREVISION_NETUID=18,
        ),
    )

    assert await bittensor_helpers.on_chain_commit_recover(element_id="element-a")
    assert json.loads(captured["data"]) == {
        "role": "miner_recover",
        "element_id": "element-a",
        "hotkey": "hk1",
    }
    assert captured["netuid"] == 18


@pytest.mark.asyncio
async def test_first_commit_block_recovers_original_block_for_tiebreak(monkeypatch):
    class RecoverySubtensor:
        async def metagraph(self, netuid, mechid=None):
            return SimpleNamespace(hotkeys=["hk1"])

    async def fake_get_subtensor():
        return RecoverySubtensor()

    async def fake_commitments(_subtensor, _netuid):
        return {
            "hk1": [
                (
                    500,
                    json.dumps(
                        {
                            "role": "miner_recover",
                            "element_id": "element-a",
                            "hotkey": "hk1",
                        }
                    ),
                )
            ]
        }

    async def fake_validator_indexes(_netuid):
        return {"validator": "https://validator.example/manako/index.json"}

    async def fake_recover(requests, _indexes):
        assert ("hk1", "element-a") in requests
        return {
            ("hk1", "element-a"): RecoveredCommitment(
                model="org/model",
                revision="rev1",
                slug="slug1",
                chute_id="chute1",
                element_id="element-a",
                commit_block=123,
                shard_block=499,
                shard_key="shard.json",
            )
        }

    monkeypatch.setattr(bittensor_helpers, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(bittensor_helpers, "get_all_revealed_commitments", fake_commitments)
    monkeypatch.setattr(
        bittensor_helpers,
        "get_settings",
        lambda: SimpleNamespace(SCOREVISION_MECHID=1),
    )
    monkeypatch.setattr(
        bittensor_helpers,
        "get_validator_indexes_from_chain",
        fake_validator_indexes,
    )
    monkeypatch.setattr(
        bittensor_helpers,
        "recover_commitments_from_shards",
        fake_recover,
    )

    blocks = await bittensor_helpers._first_commit_block_by_miner(
        18,
        element_id="element-a",
        candidate_hotkeys={"hk1"},
    )

    assert blocks == {"hk1": 123}
