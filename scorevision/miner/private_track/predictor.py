from pathlib import Path
from scorevision.miner.private_track.video import get_frame_count
from scorevision.utils.schemas import FramePrediction


def predict_actions(video_path: Path) -> list[FramePrediction]:
    frame_count = get_frame_count(video_path)

    # TODO: Replace this with your actual prediction logic
    # This example predicts "pass" on every 50th frame
    predictions = []
    for frame in range(0, frame_count, 50):
        predictions.append(FramePrediction(frame=frame, action="pass"))

    return predictions

