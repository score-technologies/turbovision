import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from scorevision.utils.manifest import Manifest
from scorevision.utils.schemas import TCGGradingPrediction
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import RegisteredMiner
from scorevision.validator.central.private_track.runner import (
    _run_challenge_for_element,
    _strip_for_public_shard,
    _trigger_scheduled_runners,
)
from scorevision.validator.central.scheduling import extract_element_tempos


MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "test_data"
    / "manifests"
    / "full_private_tcg_manifest.json"
)

PRIVATE_ELEMENT_CONFIG = {
    "manako/DetectCricketDelivery": ("cricket_delivery", "cricket_scoring", 0.3),
    "manako/TCGGrading": ("tcg_grading", "tcg_grading", 0.1),
    "manako/DetectFootballEvent": ("soccer_action", "soccer_action", 0.3),
}


def _load_manifest() -> Manifest:
    return Manifest.load_yaml(MANIFEST_PATH)


def _tcg_challenge() -> Challenge:
    return Challenge(
        challenge_id="tcg-full-manifest-1",
        image_url="https://example.com/card.png",
        ground_truth=TCGGradingPrediction(
            Header={"card_grade": 8},
            Grading_Features={
                "subgrade_surface": 7,
                "subgrade_centering": 9,
                "subgrade_edges": 8,
                "subgrade_corners": 8,
            },
        ),
        groundtruth_type="tcg_grading",
    )


def _miner() -> RegisteredMiner:
    return RegisteredMiner(
        uid=12,
        hotkey="hk-tcg",
        ip="127.0.0.1",
        port=8000,
        image_repo="org/tcg-miner",
        image_tag="v1",
        image_digest="sha256:tcg",
        commit_block=8188200,
    )


@pytest.mark.asyncio
async def test_tcg_image_is_forwarded_without_expanding_miner_contract():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "challenge_id": "tcg-full-manifest-1",
        "prediction": {
            "Header": {"card_grade": 8.9},
            "Grading_Features": {
                "subgrade_surface": 7.8,
                "subgrade_centering": 9.2,
                "subgrade_edges": 8.7,
                "subgrade_corners": 8.1,
            },
        },
        "processing_time": 1.0,
    }
    client = AsyncMock()
    client.post.return_value = response
    client_context = AsyncMock()
    client_context.__aenter__.return_value = client

    with (
        patch(
            "scorevision.validator.central.private_track.miners.httpx.AsyncClient",
            return_value=client_context,
        ),
        patch(
            "scorevision.validator.central.private_track.miners.build_signed_headers",
            return_value={"x-test": "signed"},
        ),
    ):
        attempt = await send_challenge(
            miner=_miner(),
            challenge=_tcg_challenge(),
            hotkey=object(),
            timeout=30.0,
        )

    assert attempt.timed_out is False
    assert attempt.response is not None
    request_json = client.post.await_args.kwargs["json"]
    assert "groundtruth_type" not in request_json
    assert request_json["image_url"] == "https://example.com/card.png"
    assert attempt.response.prediction.Header.card_grade == 8


@pytest.mark.asyncio
async def test_existing_video_request_keeps_legacy_wire_contract():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "challenge_id": "soccer-legacy-1",
        "prediction": {
            "type": "soccer_action",
            "items": [{"frame": 25, "action": "pass", "confidence": 1.0}],
        },
        "processing_time": 1.0,
    }
    client = AsyncMock()
    client.post.return_value = response
    client_context = AsyncMock()
    client_context.__aenter__.return_value = client
    legacy_challenge = Challenge(
        challenge_id="soccer-legacy-1",
        video_url="https://example.com/video.mp4",
        ground_truth=[],
        groundtruth_type="soccer_action",
    )

    with (
        patch(
            "scorevision.validator.central.private_track.miners.httpx.AsyncClient",
            return_value=client_context,
        ),
        patch(
            "scorevision.validator.central.private_track.miners.build_signed_headers",
            return_value={"x-test": "signed"},
        ),
    ):
        attempt = await send_challenge(
            miner=_miner(),
            challenge=legacy_challenge,
            hotkey=object(),
            timeout=30.0,
        )

    assert attempt.timed_out is False
    assert client.post.await_args.kwargs["json"] == {
        "challenge_id": "soccer-legacy-1",
        "video_url": "https://example.com/video.mp4",
        "frames": None,
    }


def test_full_manifest_loads_all_private_groundtruth_and_pillar_pairs():
    manifest = _load_manifest()

    assert len(manifest.elements) == 7
    private_elements = {
        element.id: element
        for element in manifest.elements
        if element.track == "private"
    }
    assert set(private_elements) == set(PRIVATE_ELEMENT_CONFIG)

    for element_id, (groundtruth_type, pillar, weight) in PRIVATE_ELEMENT_CONFIG.items():
        element = private_elements[element_id]
        assert element.groundtruth_type.value == groundtruth_type
        assert element.metrics is not None
        assert {
            key.value: value for key, value in element.metrics.pillars.items()
        } == {pillar: 1.0}
        assert element.weight == weight
        assert element.window_block == 300
        assert element.eval_window == 4


def test_full_manifest_private_scheduler_excludes_public_elements():
    tempos = extract_element_tempos(
        _load_manifest(),
        default_tempo=999,
        track_filter="private",
    )

    assert tempos == {
        "manako/DetectCricketDelivery": 300,
        "manako/TCGGrading": 300,
        "manako/DetectFootballEvent": 300,
    }


def test_manifest_without_tcg_keeps_only_existing_private_elements():
    raw_manifest = json.loads(MANIFEST_PATH.read_text())
    raw_manifest["elements"] = [
        element
        for element in raw_manifest["elements"]
        if element["id"] != "manako/TCGGrading"
    ]
    manifest = Manifest(**raw_manifest)

    assert extract_element_tempos(
        manifest,
        default_tempo=999,
        track_filter="private",
    ) == {
        "manako/DetectCricketDelivery": 300,
        "manako/DetectFootballEvent": 300,
    }
    assert manifest.get_element("manako/TCGGrading") is None


def test_existing_public_shard_shape_is_unchanged_without_tcg():
    result = {
        "score": 0.9,
        "challenge_id": "legacy-1",
        "miner_hotkey": "hk",
        "element_id": "manako/DetectFootballEvent",
        "groundtruth_type": "soccer_action",
    }

    assert _strip_for_public_shard(result) == {
        "score": 0.9,
        "challenge_id": "legacy-1",
        "miner_hotkey": "hk",
    }


def test_full_manifest_canonical_round_trip_keeps_tcg_configuration():
    manifest = _load_manifest()
    restored = Manifest(**json.loads(manifest.to_canonical_json()))
    tcg = restored.get_element("manako/TCGGrading")

    assert tcg is not None
    assert tcg.track == "private"
    assert tcg.groundtruth_type.value == "tcg_grading"
    assert tcg.metrics is not None
    assert {key.value: value for key, value in tcg.metrics.pillars.items()} == {
        "tcg_grading": 1.0
    }


@pytest.mark.asyncio
async def test_full_manifest_schedules_all_three_private_elements():
    manifest = _load_manifest()
    element_state = {
        element_id: {"tempo": 300, "anchor": 0, "task": None}
        for element_id in PRIVATE_ELEMENT_CONFIG
    }
    started: list[str] = []

    async def _fake_run(element_id, *_args, **_kwargs):
        started.append(element_id)

    with patch(
        "scorevision.validator.central.private_track.runner._run_challenge_for_element",
        new=AsyncMock(side_effect=_fake_run),
    ):
        _trigger_scheduled_runners(
            element_state,
            block=600,
            manifest=manifest,
            keypair=object(),
            subtensor=object(),
        )
        await asyncio.gather(*[entry["task"] for entry in element_state.values()])

    assert set(started) == set(PRIVATE_ELEMENT_CONFIG)


@pytest.mark.asyncio
async def test_full_manifest_tcg_runner_uses_manifest_type_and_pillar():
    manifest = _load_manifest()
    miner = _miner()
    settings = SimpleNamespace(
        BLACKLIST_API_URL="",
        SCOREVISION_NETUID=44,
        PRIVATE_MINER_TIMEOUT_S=30.0,
    )
    subtensor = SimpleNamespace(metagraph=AsyncMock(return_value=SimpleNamespace()))
    challenge_miner_mock = AsyncMock(
        return_value=(
            {
                "challenge_id": "tcg-full-manifest-1",
                "element_id": "manako/TCGGrading",
                "groundtruth_type": "tcg_grading",
                "miner_hotkey": miner.hotkey,
                "miner_uid": miner.uid,
                "score": 1.0,
                "prediction_count": 5,
                "ground_truth_count": 1,
                "processing_time": 1.0,
                "response_time_s": 1.0,
                "timed_out": False,
                "image_digest": miner.image_digest,
                "score_breakdown": {"tcg_grading": 1.0},
            },
            [
                {
                    "Header": {"card_grade": 8},
                    "Grading_Features": {
                        "subgrade_surface": 7,
                        "subgrade_centering": 9,
                        "subgrade_edges": 8,
                        "subgrade_corners": 8,
                    },
                }
            ],
            None,
        )
    )

    with (
        patch(
            "scorevision.validator.central.private_track.runner.get_settings",
            return_value=settings,
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_registered_miners",
            new=AsyncMock(return_value=[miner]),
        ),
        patch(
            "scorevision.validator.central.private_track.runner.get_challenge_with_ground_truth",
            new=AsyncMock(return_value=_tcg_challenge()),
        ) as challenge_mock,
        patch(
            "scorevision.validator.central.private_track.runner._challenge_miner",
            new=challenge_miner_mock,
        ),
        patch(
            "scorevision.validator.central.private_track.runner._upload_private_response_blob",
            new=AsyncMock(return_value="private_responses/tcg.json"),
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
            new=AsyncMock(),
        ),
    ):
        await _run_challenge_for_element(
            element_id="manako/TCGGrading",
            manifest=manifest,
            block=8188500,
            keypair=SimpleNamespace(ss58_address="validator-hk"),
            subtensor=subtensor,
        )

    challenge_mock.assert_awaited_once_with(
        manifest_hash=manifest.hash,
        element_id="manako/TCGGrading",
        keypair=ANY,
        groundtruth_type="tcg_grading",
    )
    args = challenge_miner_mock.await_args.args
    assert args[6] == {"tcg_grading": 1.0}
