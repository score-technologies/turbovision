import os
from typing import Any
from importlib.util import spec_from_file_location, module_from_spec
from logging import getLogger
from itertools import islice

from uvicorn import run
from fastapi import FastAPI
from PIL import Image
from huggingface_hub import snapshot_download
from ultralytics import YOLO

from scorevision.chute_template.schemas import (
    SVPredictInput,
    SVPredictOutput,
    SVFrameResult,
    SVFrame,
    SVBox,
)
from scorevision.utils.settings import get_settings
from scorevision.utils.video_processing import download_video
from scorevision.utils.image_processing import image_to_base64, pil_from_array
from scorevision.utils.async_clients import get_async_client

settings = get_settings()
# import scorevision.chute_template.load
chute_template_load_spec = spec_from_file_location(
    "chute_load",
    str(settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_LOAD_UTILS),
)
chute_template_load = module_from_spec(chute_template_load_spec)
chute_template_load.os = os
chute_template_load.Any = Any
chute_template_load.snapshot_download = snapshot_download
chute_template_load.YOLO = YOLO
chute_template_load_spec.loader.exec_module(chute_template_load)

# import scorevision.chute_template.predict
chute_template_predict_spec = spec_from_file_location(
    "chute_predict",
    str(settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_PREDICT_UTILS),
)
chute_template_predict = module_from_spec(chute_template_predict_spec)
chute_template_predict.Any = Any
chute_template_predict.Image = Image
chute_template_predict.SVFrameResult = SVFrameResult
chute_template_predict.SVPredictInput = SVPredictInput
chute_template_predict.SVPredictOutput = SVPredictOutput
chute_template_predict.SVBox = SVBox
chute_template_predict_spec.loader.exec_module(chute_template_predict)

logger = getLogger(__name__)


def deploy_mock_chute(huggingface_repo: str, huggingface_revision: str) -> None:
    chute = FastAPI(title="mock-chute")
    global model
    model = None

    @chute.on_event("startup")
    async def load_model():
        global model
        model = chute_template_load._load_model(
            repo_name=huggingface_repo,
            revision=huggingface_revision,
        )

    @chute.get("/health")
    async def health() -> dict[str, Any]:
        return chute_template_load._health(
            model=model,
            repo_name=huggingface_repo,
        )

    @chute.post("/" + settings.CHUTES_MINER_PREDICT_ENDPOINT)
    async def predict(data: SVPredictInput) -> SVPredictOutput:
        return chute_template_predict._predict(
            model=model,
            data=data,
            model_name=huggingface_repo,
        )

    @chute.get("/api/tasks/next/v2")
    async def mock_challenge():
        return {
            "task_id": "mock-challenge",
            "video_url": "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4",
        }

    run(chute)


async def test_chute_health_endpoint(base_url: str) -> None:
    logger.info("🔍 Testing `/health`...")
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"{base_url}/health"
    logger.info(url)
    try:
        async with session.get(url, headers=headers) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
            health = await response.json()
            logger.info(health)
        assert health.get("model_loaded"), "Model not loaded"
        logger.info("✅ /health passed")
    except Exception as e:
        logger.error(f"❌ /health failed: {e}")


async def get_chute_logs(instance_id: str) -> None:
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"https://api.chutes.ai/instances/{instance_id}/logs"  # ?backfill=10000"
    logger.info(url)
    try:
        async with session.get(url, headers=headers) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
    except Exception as e:
        logger.error(f"❌ /logs failed: {e}")


async def test_chute_predict_endpoint(
    base_url: str, video_url: str, first_n_frames: int
) -> None:
    logger.info("🔍 Testing `/predict`...")
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}"
    logger.info(url)
    try:
        _, frames, _ = await download_video(url=video_url, frame_numbers=[])
        logger.info(f"{len(frames)} frames extracted from video")
        frames = dict(islice(frames.items(), first_n_frames))
        logger.info(f"Testing on first {len(frames)} frames")
        b64_frames = [
            SVFrame(
                frame_id=frame_number,
                data=image_to_base64(
                    img=pil_from_array(array=frame),
                    fmt="JPEG",
                    quality=settings.SCOREVISION_IMAGE_JPEG_QUALITY,
                    optimise=True,
                ),
            )
            for frame_number, frame in frames.items()
        ]
        logger.info("Frames converted to b64")
        payload = payload = SVPredictInput(
            frames=b64_frames[:first_n_frames], meta=None
        )
        async with session.post(
            url,
            headers=headers,
            json=payload.model_dump(mode="json"),
        ) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
            assert response.status == 200, "Non-200 response from predict"
            output = await response.json()
            logger.info(output)
        assert output["success"] is True, f"Prediction failed: {output}"
        predictions = output.get("predictions", {})
        assert "frames" in predictions, "Missing predictions"
        n_predictions = len(predictions["frames"])
        assert n_predictions == len(
            frames
        ), f"Number of predictions returned ({n_predictions}) does not match Number of frames given ({len(frames)})"
        frame_ids = set(frame["frame_id"] for frame in predictions["frames"])
        assert len(frame_ids) == len(
            frames
        ), f"Number of unique frame ids returned {len(frame_ids)} does not match number of unique frames given ({len(frames)})"
        logger.info("✅ /predict passed")
    except Exception as e:
        logger.error(f"❌ /predict failed: {e}")
