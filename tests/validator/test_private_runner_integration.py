import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scorevision.utils.manifest import Element, Manifest, Metrics, PillarName, Tee
from scorevision.utils.schemas import ChallengeResponse, CricketDeliveryPrediction, FramePrediction
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner
from scorevision.validator.central.private_track.runner import (
    _run_challenge_for_element,
    _trigger_scheduled_runners,
)


def _private_manifest() -> Manifest:
    soccer = Element(
        id="manako/DetectFootballEvent",
        track="private",
        weight=0.2,
        window_block=300,
        eval_window=4,
        beta=1.0,
        groundtruth_type="soccer_action",
        metrics=Metrics(pillars={PillarName.SOCCER_ACTION: 1.0}),
    )
    cricket = Element(
        id="manako/DetectCricketDelivery",
        track="private",
        weight=0.2,
        window_block=300,
        eval_window=4,
        beta=1.0,
        groundtruth_type="cricket_delivery",
        metrics=Metrics(pillars={PillarName.CRICKET_SCORING: 1.0}),
    )
    return Manifest(
        window_id="2025-10-27",
        version=1.3,
        expiry_block=12345678910,
        tee=Tee(trusted_share_gamma=0.2),
        elements=[soccer, cricket],
    )


def _miner(uid: int, hotkey: str) -> RegisteredMiner:
    return RegisteredMiner(
        uid=uid,
        hotkey=hotkey,
        ip="127.0.0.1",
        port=8000,
        image_repo="org/private-miner",
        image_tag="v1",
        image_digest=f"sha256:{uid}",
        commit_block=123,
    )


def _soccer_challenge() -> Challenge:
    return Challenge(
        challenge_id="soccer-1",
        video_url="https://example.com/soccer.mp4",
        ground_truth=[FramePrediction(frame=25, action="pass")],
        groundtruth_type="soccer_action",
    )


def _cricket_challenge() -> Challenge:
    return Challenge(
        challenge_id="cricket-1",
        video_url="https://example.com/cricket.mp4",
        ground_truth=CricketDeliveryPrediction(kph=128.2, bounce_x=6.0, stump_y=0.1),
        groundtruth_type="cricket_delivery",
    )


@pytest.mark.asyncio
async def test_trigger_scheduled_runners_launches_two_private_elements_in_parallel():
    manifest = _private_manifest()
    block = 600
    keypair = object()
    subtensor = object()
    element_state = {
        "manako/DetectFootballEvent": {"tempo": 300, "anchor": 0, "task": None},
        "manako/DetectCricketDelivery": {"tempo": 300, "anchor": 0, "task": None},
    }

    started: list[str] = []

    async def _fake_run(element_id, *_args, **_kwargs):
        started.append(element_id)

    with patch(
        "scorevision.validator.central.private_track.runner._run_challenge_for_element",
        new=AsyncMock(side_effect=_fake_run),
    ):
        _trigger_scheduled_runners(element_state, block, manifest, keypair, subtensor)
        await asyncio.gather(*[entry["task"] for entry in element_state.values()])

    assert sorted(started) == sorted(
        ["manako/DetectFootballEvent", "manako/DetectCricketDelivery"]
    )


@pytest.mark.asyncio
async def test_run_challenge_for_element_soccer_uses_soccer_pillars_and_uploads_results():
    manifest = _private_manifest()
    miner = _miner(7, "hk-soccer")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))

    challenge_miner_mock = AsyncMock(
        return_value=(
            {
                "challenge_id": "soccer-1",
                "element_id": "manako/DetectFootballEvent",
                "miner_hotkey": miner.hotkey,
                "miner_uid": miner.uid,
                "score": 0.91,
                "prediction_count": 1,
                "ground_truth_count": 1,
                "processing_time": 1.2,
                "response_time_s": 1.2,
                "timed_out": False,
                "image_digest": miner.image_digest,
                "score_breakdown": {"soccer_action": 0.91},
            },
            [{"frame": 25, "action": "pass", "confidence": 1.0}],
            None,
        )
    )
    upload_shard_mock = AsyncMock(return_value="privatevision_results/shard.json")

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=_soccer_challenge()),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._challenge_miner",
            new=challenge_miner_mock,
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_private_response_blob",
            new=AsyncMock(return_value="private_responses/key.json"),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._emit_private_score_to_public_db",
            new=AsyncMock(),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_benchmark_result",
            new=AsyncMock(),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=upload_shard_mock,
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectFootballEvent",
            manifest=manifest,
            block=9000,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert challenge_miner_mock.await_count == 1
    args = challenge_miner_mock.await_args.args
    assert args[6] == {"soccer_action": 1.0}
    assert upload_shard_mock.await_count == 1


@pytest.mark.asyncio
async def test_run_challenge_for_element_cricket_uses_cricket_pillars_and_uploads_results():
    manifest = _private_manifest()
    miner = _miner(8, "hk-cricket")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))

    challenge_miner_mock = AsyncMock(
        return_value=(
            {
                "challenge_id": "cricket-1",
                "element_id": "manako/DetectCricketDelivery",
                "miner_hotkey": miner.hotkey,
                "miner_uid": miner.uid,
                "score": 0.77,
                "prediction_count": 1,
                "ground_truth_count": 1,
                "processing_time": 2.1,
                "response_time_s": 2.1,
                "timed_out": False,
                "image_digest": miner.image_digest,
                "score_breakdown": {"cricket_scoring": 0.77},
            },
            [{"kph": 128.2, "bounce_x": 6.0, "stump_y": 0.1}],
            None,
        )
    )
    upload_shard_mock = AsyncMock(return_value="privatevision_results/shard.json")

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=_cricket_challenge()),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._challenge_miner",
            new=challenge_miner_mock,
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_private_response_blob",
            new=AsyncMock(return_value="private_responses/key.json"),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._emit_private_score_to_public_db",
            new=AsyncMock(),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_benchmark_result",
            new=AsyncMock(),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=upload_shard_mock,
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectCricketDelivery",
            manifest=manifest,
            block=9001,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert challenge_miner_mock.await_count == 1
    args = challenge_miner_mock.await_args.args
    assert args[6] == {"cricket_scoring": 1.0}
    assert upload_shard_mock.await_count == 1


@pytest.mark.asyncio
async def test_run_challenge_for_element_skips_when_no_registered_miners():
    manifest = _private_manifest()
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=AsyncMock(),
        ) as upload_shard_mock,
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectFootballEvent",
            manifest=manifest,
            block=9002,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert upload_shard_mock.await_count == 0


@pytest.mark.asyncio
async def test_run_challenge_for_element_skips_when_no_valid_challenge():
    manifest = _private_manifest()
    miner = _miner(9, "hk-no-challenge")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=AsyncMock(),
        ) as upload_shard_mock,
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectFootballEvent",
            manifest=manifest,
            block=9003,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert upload_shard_mock.await_count == 0


@pytest.mark.asyncio
async def test_run_challenge_for_element_mixed_timeout_and_success_are_both_sharded():
    manifest = _private_manifest()
    miner_ok = _miner(10, "hk-ok")
    miner_timeout = _miner(11, "hk-timeout")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))

    challenge_miner_mock = AsyncMock(
        side_effect=[
            (
                {
                    "challenge_id": "soccer-1",
                    "element_id": "manako/DetectFootballEvent",
                    "miner_hotkey": miner_ok.hotkey,
                    "miner_uid": miner_ok.uid,
                    "score": 0.93,
                    "prediction_count": 1,
                    "ground_truth_count": 1,
                    "processing_time": 1.0,
                    "response_time_s": 1.0,
                    "timed_out": False,
                    "image_digest": miner_ok.image_digest,
                    "score_breakdown": {"soccer_action": 0.93},
                },
                [{"frame": 25, "action": "pass", "confidence": 1.0}],
                None,
            ),
            (
                {
                    "challenge_id": "soccer-1",
                    "element_id": "manako/DetectFootballEvent",
                    "miner_hotkey": miner_timeout.hotkey,
                    "miner_uid": miner_timeout.uid,
                    "score": 0.0,
                    "prediction_count": 0,
                    "ground_truth_count": 1,
                    "processing_time": 30.2,
                    "response_time_s": 30.2,
                    "timed_out": True,
                    "image_digest": miner_timeout.image_digest,
                    "score_breakdown": {},
                },
                None,
                None,
            ),
        ]
    )
    upload_shard_mock = AsyncMock(return_value="privatevision_results/shard.json")

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner_ok, miner_timeout]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=_soccer_challenge()),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._challenge_miner",
            new=challenge_miner_mock,
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_private_response_blob",
            new=AsyncMock(return_value="private_responses/key.json"),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._emit_private_score_to_public_db",
            new=AsyncMock(),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=upload_shard_mock,
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectFootballEvent",
            manifest=manifest,
            block=9004,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert upload_shard_mock.await_count == 1
    shard_args = upload_shard_mock.await_args.args
    results = shard_args[0]
    assert len(results) == 2
    assert any(r["timed_out"] is True for r in results)
    assert any(r["timed_out"] is False for r in results)


@pytest.mark.asyncio
async def test_run_challenge_for_element_calls_registry_with_selected_element_id():
    manifest = _private_manifest()
    miner = _miner(12, "hk-filter")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))
    get_registered_miners_mock = AsyncMock(return_value=[miner])

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=get_registered_miners_mock,
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=None),
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectCricketDelivery",
            manifest=manifest,
            block=9005,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert get_registered_miners_mock.await_count == 1
    assert (
        get_registered_miners_mock.await_args.kwargs["element_id"]
        == "manako/DetectCricketDelivery"
    )


@pytest.mark.asyncio
async def test_run_challenge_for_element_continues_when_emit_private_score_fails():
    manifest = _private_manifest()
    miner = _miner(13, "hk-emit-fail")
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))
    upload_shard_mock = AsyncMock(return_value="privatevision_results/shard.json")

    with (
        patch("scorevision.validator.central.private_track.runner.get_settings", return_value=settings),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=_soccer_challenge()),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._challenge_miner",
            new=AsyncMock(
                return_value=(
                    {
                        "challenge_id": "soccer-1",
                        "element_id": "manako/DetectFootballEvent",
                        "miner_hotkey": miner.hotkey,
                        "miner_uid": miner.uid,
                        "score": 0.88,
                        "prediction_count": 1,
                        "ground_truth_count": 1,
                        "processing_time": 1.4,
                        "response_time_s": 1.4,
                        "timed_out": False,
                        "image_digest": miner.image_digest,
                        "score_breakdown": {"soccer_action": 0.88},
                    },
                    [{"frame": 25, "action": "pass", "confidence": 1.0}],
                    None,
                )
            ),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_private_response_blob",
            new=AsyncMock(return_value="private_responses/key.json"),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._emit_private_score_to_public_db",
            new=AsyncMock(side_effect=RuntimeError("emit failed")),
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_shard",
            new=upload_shard_mock,
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/DetectFootballEvent",
            manifest=manifest,
            block=9006,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    assert upload_shard_mock.await_count == 1
