from unittest.mock import AsyncMock

import pytest

from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction
from scorevision.validator.audit.private_track import spotcheck as spotcheck_mod


def _cricket_prediction_dict() -> dict:
    return {
        "match": "Sri Lanka v India",
        "matchid": 34429,
        "inningsid": 1,
        "overid": 2,
        "ball_in_over": 3,
        "ballid": 9,
        "xlsx_overs": "2.03",
        "scorecard_overs": "1.3",
        "kph": 126.5,
        "release_y": -0.49,
        "release_z": 1.94,
        "bounce_x": 8.0,
        "bounce_y": -0.19,
        "impact_x": 1.7,
        "impact_y": -0.03,
        "impact_z": 0.89,
        "interception_distance": 6.27,
        "stump_y": 0.02,
        "deviation": 1.0,
        "swing_angle": -2.0,
        "stump_z": 1.04,
        "runs": 0,
        "wickets": 0,
    }


def test_rescore_miner_uses_cricket_scorer(monkeypatch):
    ground_truth = CricketDeliveryPrediction(**_cricket_prediction_dict())
    calls = []

    def fake_score(prediction, actual):
        calls.append((prediction, actual))
        return 0.64, {"kph": 1.0}

    monkeypatch.setattr(spotcheck_mod, "score_cricket_prediction_with_breakdown", fake_score)

    score = spotcheck_mod.rescore_miner({"prediction": _cricket_prediction_dict()}, ground_truth)

    assert score == 0.64
    prediction, actual = calls[0]
    assert isinstance(prediction, CricketDeliveryPrediction)
    assert actual is ground_truth


def test_rescore_miner_accepts_existing_cricket_singleton_predictions_blob():
    ground_truth = CricketDeliveryPrediction(**_cricket_prediction_dict())

    score = spotcheck_mod.rescore_miner({"predictions": [_cricket_prediction_dict()]}, ground_truth)

    assert score == 1.0


def test_rescore_miner_uses_legacy_scorer(monkeypatch):
    ground_truth = [FramePrediction(frame=25, action="pass")]
    calls = []

    def fake_score(predictions, actual):
        calls.append((predictions, actual))
        return 0.83

    monkeypatch.setattr(spotcheck_mod, "score_predictions", fake_score)

    score = spotcheck_mod.rescore_miner(
        {"predictions": [{"frame": 25, "action": "pass", "confidence": 0.7}]},
        ground_truth,
    )

    assert score == 0.83
    predictions, actual = calls[0]
    assert predictions == [FramePrediction(frame=25, action="pass", confidence=0.7)]
    assert actual is ground_truth


@pytest.mark.asyncio
async def test_fetch_miner_responses_accepts_prediction_and_predictions(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "responses": [
                    {
                        "miner_hotkey": "legacy-hotkey",
                        "predictions": [{"frame": 25, "action": "pass"}],
                    },
                    {
                        "miner_hotkey": "cricket-hotkey",
                        "prediction": _cricket_prediction_dict(),
                    },
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers):
            assert url == "https://responses.example/api/private-track/responses/123"
            assert headers == {}
            return FakeResponse()

    monkeypatch.setattr(
        spotcheck_mod,
        "get_settings",
        lambda: type("Settings", (), {"PRIVATE_MINER_RESPONSES_API_URL": "https://responses.example"})(),
    )
    monkeypatch.setattr(spotcheck_mod, "build_signed_headers", lambda keypair: {})
    monkeypatch.setattr(spotcheck_mod.httpx, "AsyncClient", lambda timeout: FakeClient())

    responses = await spotcheck_mod.fetch_miner_responses("123", keypair=None)

    assert responses["legacy-hotkey"] == [{"frame": 25, "action": "pass"}]
    assert responses["cricket-hotkey"] == _cricket_prediction_dict()


@pytest.mark.asyncio
async def test_run_private_spotcheck_propagates_element_id(monkeypatch):
    fetch_ground_truth = AsyncMock(return_value=[FramePrediction(frame=25, action="pass")])
    monkeypatch.setattr(spotcheck_mod, "fetch_ground_truth", fetch_ground_truth)
    monkeypatch.setattr(
        spotcheck_mod,
        "fetch_miner_responses",
        AsyncMock(return_value={"miner-hotkey": [{"frame": 25, "action": "pass"}]}),
    )
    monkeypatch.setattr(spotcheck_mod, "score_predictions", lambda predictions, ground_truth: 0.9)

    results = await spotcheck_mod.run_private_spotcheck(
        "123",
        [
            {
                "miner_hotkey": "miner-hotkey",
                "element_id": "manak0/Element-CricketBallTrack",
                "score": 0.9,
            }
        ],
        keypair=None,
        threshold=0.99,
    )

    fetch_ground_truth.assert_awaited_once_with(
        "123",
        None,
        element_id="manak0/Element-CricketBallTrack",
    )
    assert len(results) == 1
    assert results[0].element_id == "manak0/Element-CricketBallTrack"
    assert results[0].passed is True
