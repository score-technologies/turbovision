from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import bittensor as bt
import pytest
from bittensor.core.types import ExtrinsicResponse

from scorevision.utils import bittensor_helpers
from scorevision.validator.core import signer
from scorevision.validator.core import weights as weights_module
from scorevision.validator.central.open_source import runner as public_runner
from scorevision.validator.central.private_track import runner as private_runner


def test_bittensor_v10_public_api_is_available():
    assert bt.Wallet is not None
    assert bt.Subtensor is not None
    assert bt.AsyncSubtensor is not None


@pytest.mark.asyncio
async def test_get_subtensor_uses_v10_async_constructor(monkeypatch):
    created = []

    class FakeAsyncSubtensor:
        def __init__(self, *, network):
            self.network = network
            self.initialize = AsyncMock(return_value=self)
            created.append(self)

    monkeypatch.setattr(bittensor_helpers, "AsyncSubtensor", FakeAsyncSubtensor)
    monkeypatch.setattr(
        bittensor_helpers,
        "get_settings",
        lambda: SimpleNamespace(BITTENSOR_SUBTENSOR_ENDPOINT="wss://chain.example"),
    )
    monkeypatch.setattr(bittensor_helpers, "_SUBTENSOR", None)

    result = await bittensor_helpers.get_subtensor()

    assert result is created[0]
    assert result.network == "wss://chain.example"
    result.initialize.assert_awaited_once_with()


def test_signer_submits_weights_with_v10_extrinsic_response(monkeypatch):
    response = ExtrinsicResponse(success=True, message="included")
    subtensor = SimpleNamespace(set_weights=Mock(return_value=response))
    monkeypatch.setattr(signer, "_get_sync_subtensor", lambda: subtensor)

    ok = signer._set_weights(
        wallet=object(),
        netuid=44,
        mechid=1,
        uids=[2, 7],
        weights=[0.25, 0.75],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert ok is True
    subtensor.set_weights.assert_called_once_with(
        wallet=ANY,
        netuid=44,
        mechid=1,
        uids=[2, 7],
        weights=[0.25, 0.75],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def test_signer_preserves_setting_weights_too_fast_semantics(monkeypatch):
    response = ExtrinsicResponse(success=False, message="SettingWeightsTooFast")
    monkeypatch.setattr(
        signer,
        "_get_sync_subtensor",
        lambda: SimpleNamespace(set_weights=Mock(return_value=response)),
    )

    assert signer._set_weights(
        wallet=object(),
        netuid=44,
        mechid=1,
        uids=[2],
        weights=[1.0],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ) is True


def test_sign_payloads_produces_verifiable_hotkey_signatures():
    hotkey = bt.Keypair.create_from_uri("//Alice")
    wallet = SimpleNamespace(hotkey=hotkey)
    payloads = ["nonce-1", "unicode-payload-é"]

    signatures = signer._sign_payloads(wallet, payloads)

    assert len(signatures) == len(payloads)
    for payload, signature in zip(payloads, signatures):
        assert hotkey.verify(payload.encode("utf-8"), bytes.fromhex(signature))


class _OneBlockSubtensor:
    def __init__(self, shutdown_event):
        self.shutdown_event = shutdown_event
        self.get_current_block = AsyncMock(side_effect=self._get_current_block)
        self.wait_for_block = AsyncMock(return_value=True)

    async def _get_current_block(self):
        self.shutdown_event.set()
        return 1


@pytest.mark.asyncio
async def test_weight_loop_smoke_with_mocked_v10_subtensor(monkeypatch):
    weights_module.shutdown_event.clear()
    subtensor = _OneBlockSubtensor(weights_module.shutdown_event)
    settings = SimpleNamespace(
        SCOREVISION_NETUID=44,
        VALIDATOR_FALLBACK_UID=0,
        VALIDATOR_TAIL_BLOCKS=100,
        BITTENSOR_WALLET_COLD="cold",
        BITTENSOR_WALLET_HOT="hot",
        SCOREVISION_CENTRAL_VALIDATOR_HOTKEY="validator-hk",
        BLACKLIST_API_URL="",
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator-hk"))

    monkeypatch.setattr(weights_module, "get_settings", lambda: settings)
    monkeypatch.setattr(weights_module.bt, "Wallet", lambda **_kwargs: wallet)
    monkeypatch.setattr(weights_module, "get_validator_hotkey_ss58", lambda: "validator-hk")
    monkeypatch.setattr(weights_module, "setup_signal_handler", lambda: None)
    monkeypatch.setattr(weights_module, "get_subtensor", AsyncMock(return_value=subtensor))
    monkeypatch.setattr(weights_module, "fetch_blacklisted_hotkeys", AsyncMock(return_value=set()))
    monkeypatch.setattr(weights_module, "fetch_compliance_failure_tuples", AsyncMock(return_value=set()))

    await weights_module.weights_loop(tempo=150)

    subtensor.get_current_block.assert_awaited_once_with()
    subtensor.wait_for_block.assert_awaited_once_with()
    weights_module.shutdown_event.clear()


@pytest.mark.asyncio
async def test_public_runner_loop_smoke_with_mocked_v10_subtensor(monkeypatch):
    public_runner.shutdown_event.clear()
    subtensor = _OneBlockSubtensor(public_runner.shutdown_event)
    settings = SimpleNamespace(
        SCOREVISION_NETUID=44,
        RUNNER_GET_BLOCK_TIMEOUT_S=1.0,
        RUNNER_WAIT_BLOCK_TIMEOUT_S=1.0,
        RUNNER_RECONNECT_DELAY_S=0.0,
        RUNNER_DEFAULT_ELEMENT_TEMPO=300,
    )
    manifest = SimpleNamespace(hash="manifest-hash")

    monkeypatch.setattr(public_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(public_runner, "setup_shutdown_handler", lambda _event: None)
    monkeypatch.setattr(public_runner, "_commit_central_validator_on_start", AsyncMock())
    monkeypatch.setattr(public_runner, "get_subtensor", AsyncMock(return_value=subtensor))
    monkeypatch.setattr(public_runner, "load_manifest", AsyncMock(return_value=manifest))
    monkeypatch.setattr(public_runner, "extract_element_tempos", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(public_runner, "close_http_clients_async", AsyncMock())

    await public_runner.runner_loop()

    subtensor.get_current_block.assert_awaited_once_with()
    subtensor.wait_for_block.assert_awaited_once_with()
    public_runner.shutdown_event.clear()


@pytest.mark.asyncio
async def test_private_runner_loop_smoke_with_mocked_v10_subtensor(monkeypatch):
    private_runner.shutdown_event.clear()
    subtensor = _OneBlockSubtensor(private_runner.shutdown_event)
    settings = SimpleNamespace(
        SCOREVISION_NETUID=44,
        BITTENSOR_WALLET_COLD="cold",
        BITTENSOR_WALLET_HOT="hot",
        RUNNER_GET_BLOCK_TIMEOUT_S=1.0,
        RUNNER_WAIT_BLOCK_TIMEOUT_S=1.0,
        RUNNER_RECONNECT_DELAY_S=0.0,
        RUNNER_DEFAULT_ELEMENT_TEMPO=300,
    )
    manifest = SimpleNamespace(hash="manifest-hash")

    monkeypatch.setattr(private_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(private_runner, "setup_shutdown_handler", lambda _event: None)
    monkeypatch.setattr(private_runner, "load_hotkey_keypair", lambda *_args: object())
    monkeypatch.setattr(private_runner, "get_subtensor", AsyncMock(return_value=subtensor))
    monkeypatch.setattr(private_runner, "load_manifest", AsyncMock(return_value=manifest))
    monkeypatch.setattr(private_runner, "extract_element_tempos", lambda *_args, **_kwargs: {})

    await private_runner.challenge_loop()

    subtensor.get_current_block.assert_awaited_once_with()
    subtensor.wait_for_block.assert_awaited_once_with()
    private_runner.shutdown_event.clear()
