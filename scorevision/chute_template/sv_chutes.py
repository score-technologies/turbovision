#!/usr/bin/env python3

# Input/Output schemas for ScoreVision

from typing import Any
from base64 import b64decode
from io import BytesIO

from pydantic import BaseModel
from PIL import Image


class SVFrame(BaseModel):
    frame_id: int
    data: str  # base64 encoded image

    @property
    def image(self) -> Image.Image:
        return Image.open(BytesIO(b64decode(self.data))).convert("RGB")


class SVPredictInput(BaseModel):
    frames: list[SVFrame]
    meta: dict[str, Any] | None = None


class SVBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls: str = "player"
    conf: float = 1.0


class SVFrameResult(BaseModel):
    frame_id: int
    boxes: list[SVBox]


class SVPredictOutput(BaseModel):
    success: bool
    model: str
    predictions: dict[str, list[SVFrameResult]] | None = None
    error: str | None = None


import os

from chutes.chute import Chute, NodeSelector
from chutes.image import Image as ChutesImage

model = None
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def init_chute(username: str, name: str) -> Chute:
    image = (
        ChutesImage(
            username=username,
            name=name,
            tag="latest",
        )
        .from_base("python:3.12-bullseye")
        .run_command("python -m pip install --upgrade pip setuptools wheel")
        .run_command(
            """pip install --no-cache-dir \
            pillow==10.0.1 \
            huggingface_hub==0.19.4"""
        )
        # .run_command("pip install ultralytics==8.0.206")  # YOLO support
        .set_workdir("/app")
    )

    node_selector = NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
    )
    return Chute(
        username=username,
        name=name,
        image=image,
        node_selector=node_selector,
        concurrency=4,
        timeout_seconds=300,
    )


import os
from huggingface_hub import snapshot_download

# from ultralytics import YOLO


def _health(model: Any | None, repo_name: str) -> dict[str, Any]:
    return {
        "status": "healthy",
        "model": repo_name,
        "model_loaded": model is not None,
    }


def load_model_from_huggingface_hub(model_path: str):
    pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
    if pt_files:
        model_file = os.path.join(model_path, pt_files[0])
        model = True  # eg. model = YOLO(model_file)

        print(f"Loaded YOLO model: {pt_files[0]}")
        return model
    raise ValueError("No .pt file found for YOLO model")


def _load_model(repo_name: str, revision: str):
    try:
        model_path = snapshot_download(repo_name, revision=revision)
        print(f"Downloaded model from Hf to: {model_path}")

        return load_model_from_huggingface_hub(model_path=model_path)

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def model_predict(model, images: list[Image.Image]) -> list[SVFrameResult]:
    frame_results = []
    for i in range(750):
        frame_results.append(
            SVFrameResult(
                frame_id=i,
                boxes=[
                    SVBox(
                        x1=10,
                        y1=20,
                        x2=50,
                        y2=33,
                        cls="player",
                        conf=0.0,
                    )
                ],
            )
        )
    # ---- example using YOLO------
    # detections = model(images)
    # for i, detection in enumerate(detections):
    #     boxes = []
    #     if hasattr(detection, "boxes") and detection.boxes is not None:
    #         for box in detection.boxes.data:
    #             x1, y1, x2, y2, conf, cls = box.tolist()
    #             boxes.append(
    #                 SVBox(
    #                     x1=int(x1),
    #                     y1=int(y1),
    #                     x2=int(x2),
    #                     y2=int(y2),
    #                     cls="player",
    #                     conf=float(conf),
    #                 )
    #             )
    #         frame_results.append(SVFrameResult(frame_id=i, boxes=boxes))
    return frame_results


def _predict(
    model: Any | None, data: SVPredictInput, model_name: str
) -> SVPredictOutput:
    try:
        if not model:
            return SVPredictOutput(
                success=False, error="Model not loaded", model=model_name
            )

        if not data.frames:
            return SVPredictOutput(
                success=False, error="No frames provided", model=model_name
            )

        images = []
        for frame in data.frames:
            try:
                images.append(frame.image)
            except Exception as e:
                return SVPredictOutput(
                    success=False,
                    error=f"Failed to decode frame {frame.frame_id}: {str(e)}",
                    model=model_name,
                )

        frame_results = model_predict(model=model, images=images)

        return SVPredictOutput(
            success=True, model=model_name, predictions={"frames": frame_results}
        )

    except Exception as e:
        from traceback import format_exc

        print(f"Error in predict_scorevision: {str(e)}")
        print(format_exc())

        return SVPredictOutput(success=False, error=str(e), model=model_name)


from typing import Any

chute = init_chute(
    username="score_test",
    name="mterryjack-boss-glider",
)

@chute.on_startup()
async def load_model(self):    
    global model
    model = _load_model(
        repo_name="MTerryJack/ScoreVision-my-hotkey", 
        revision="c564831cfb1756f59ef442ccbc6f4db716be8706", 
    )

@chute.cord(public_api_path="/health", method="GET", public_api_method="GET")
async def health(self) -> dict[str, Any]:
    return _health(
        model=model, 
        repo_name="mterryjack-boss-glider", 
    )


@chute.cord(
    public_api_path="/predict",
    method="POST",
    public_api_method="POST"
)
async def predict(self, data: SVPredictInput) -> SVPredictOutput:
    return _predict(
        model=model,
        data=data, 
        model_name= "mterryjack-boss-glider", 
    )