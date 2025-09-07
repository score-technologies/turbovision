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
        model = YOLO(model_file)
        print(f"Loaded YOLO model: {pt_files[0]}")
        return model
    raise ValueError("No .pt file found for YOLO model")


def _load_model(repo_name: str, revision: str):
    try:
        model_path = snapshot_download(repo_name, revision=revision)
        print(f"Downloaded model from Hf to: {model_path}")
        model = load_model_from_huggingface_hub(model_path=model_path)
        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
