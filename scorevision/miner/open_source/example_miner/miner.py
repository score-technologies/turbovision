from pathlib import Path

from ultralytics import YOLO
from numpy import ndarray
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class Miner:
    """
    This class is responsible for:
    - Loading ML models.
    - Running batched predictions on images.
    - Parsing ML model outputs into structured results (TVFrameResult).

    This class can be modified, but it must have the following to be compatible with the chute:
        - be named `Miner`
        - have a `predict_batch` function with the inputs and outputs specified
        - be stored in a file called `miner.py` which lives in the root of the HFHub repo
    """

    def __init__(self, path_hf_repo: Path) -> None:
        """
        Loads all ML models from the repository.
        -----(Adjust as needed)----

        Args:
            path_hf_repo (Path):
                Path to the downloaded HuggingFace Hub repository

        Returns:
            None
        """
        self.bbox_model = YOLO(path_hf_repo / "football-player-detection.pt")
        print(f"✅ BBox Model Loaded")
        self.keypoints_model = YOLO(path_hf_repo / "football-pitch-detection.pt")
        print(f"✅ Keypoints Model Loaded")

    def __repr__(self) -> str:
        """
        Information about miner returned in the health endpoint
        to inspect the loaded ML models (and their types)
        -----(Adjust as needed)----
        """
        return f"BBox Model: {type(self.bbox_model).__name__}\nKeypoints Model: {type(self.keypoints_model).__name__}"

    def predict_batch(
        self,
        batch_images: list[ndarray],
        offset: int,
        n_keypoints: int,
    ) -> list[TVFrameResult]:
        """
        Miner prediction for a batch of images.
        Handles the orchestration of ML models and any preprocessing and postprocessing
        -----(Adjust as needed)----

        Args:
            batch_images (list[np.ndarray]):
                A list of images (as NumPy arrays) to process in this batch.
            offset (int):
                The frame number corresponding to the first image in the batch.
                Used to correctly index frames in the output results.
            n_keypoints (int):
                The number of keypoints expected for each frame in this challenge type.

        Returns:
            list[TVFrameResult]:
                A list of predictions for each image in the batch
        """

        bboxes: dict[int, list[BoundingBox]] = {}
        bbox_model_results = self.bbox_model.predict(batch_images)
        if bbox_model_results is not None:
            for frame_number_in_batch, detection in enumerate(bbox_model_results):
                if not hasattr(detection, "boxes") or detection.boxes is None:
                    continue
                boxes = []
                for box in detection.boxes.data:
                    x1, y1, x2, y2, conf, cls_id = box.tolist()
                    boxes.append(
                        BoundingBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            cls_id=int(cls_id),
                            conf=float(conf),
                        )
                    )
                bboxes[offset + frame_number_in_batch] = boxes
        print("✅ BBoxes predicted")

        keypoints: dict[int, tuple[int, int]] = {}
        keypoints_model_results = self.keypoints_model.predict(batch_images)
        if keypoints_model_results is not None:
            for frame_number_in_batch, detection in enumerate(keypoints_model_results):
                if not hasattr(detection, "keypoints") or detection.keypoints is None:
                    continue
                frame_keypoints: list[tuple[int, int]] = []
                for part_points in detection.keypoints.data:
                    for x, y, _ in part_points:
                        frame_keypoints.append((int(x), int(y)))
                if len(frame_keypoints) < n_keypoints:
                    frame_keypoints.extend(
                        [(0, 0)] * (n_keypoints - len(frame_keypoints))
                    )
                else:
                    frame_keypoints = frame_keypoints[:n_keypoints]
                keypoints[offset + frame_number_in_batch] = frame_keypoints
        print("✅ Keypoints predicted")

        results: list[TVFrameResult] = []
        for frame_number in range(offset, offset + len(batch_images)):
            results.append(
                TVFrameResult(
                    frame_id=frame_number,
                    boxes=bboxes.get(frame_number, []),
                    keypoints=keypoints.get(
                        frame_number, [(0, 0) for _ in range(n_keypoints)]
                    ),
                )
            )
        print("✅ Combined results as TVFrameResult")
        return results
