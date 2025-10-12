def init_chute(username: str, name: str) -> Chute:
    image = (
        ChutesImage(
            username=username,
            name=name,
            tag="latest",
        )
        .from_base("parachutes/python:3.12")
        .run_command("pip install --upgrade setuptools wheel")
        .run_command(
            "pip install pillow==10.0.1 huggingface_hub==0.19.4 ultralytics==8.0.206 'torch<2.6' opencv-python-headless"
        )
        .set_workdir("/app")
    )

    node_selector = NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        include=["a100", "a100_40gb", "3090", "a40", "a6000"],
        exclude=["5090", "b200", "h200", "h20", "mi300x"],
    )
    return Chute(
        username=username,
        name=name,
        image=image,
        node_selector=node_selector,
        concurrency=4,
        timeout_seconds=300,
        max_instances=5,
        scaling_threshold=0.5,
    )
