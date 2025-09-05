import os
from huggingface_hub import snapshot_download
import torch


def health(model: Any | None, repo_name: str) -> dict[str, Any]:
    return {
        "status": "healthy",
        "model": repo_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": model is not None,
    }


def load_ultralytics_model(model_path: str):
    from ultralytics import YOLO

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        return load_ultralytics_model(model_path=model_path)

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
