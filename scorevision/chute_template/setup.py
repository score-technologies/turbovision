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
        .from_base("parachutes/python:3.12")
        .run_command(
            """pip install \
            opencv-python==4.8.0.76 \
            pillow==10.0.1 \
            numpy>=1.25.0 \ 
            huggingface_hub==0.19.4"""
        )
        .run_command("pip install ultralytics==8.0.206")  # YOLO support
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
