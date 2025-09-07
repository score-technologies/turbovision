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
            "pip install pillow==10.0.1 huggingface_hub==0.19.4 ultralytics==8.0.206 'torch<2.6'"
        )
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
