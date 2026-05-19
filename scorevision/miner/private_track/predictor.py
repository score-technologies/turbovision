from pathlib import Path
from scorevision.miner.private_track.video import get_frame_count
from scorevision.utils.schemas import ChallengeRequest, CricketDeliveryPrediction, FramePrediction


def predict_actions(video_path: Path) -> list[FramePrediction]:
    frame_count = get_frame_count(video_path)

    # TODO: Replace this with your actual prediction logic
    # This example predicts "pass" on every 25th frame
    predictions = []
    for frame in range(0, frame_count, 25):
        predictions.append(FramePrediction(frame=frame, action="pass"))

    return predictions


def predict_cricket_delivery(request: ChallengeRequest) -> CricketDeliveryPrediction:
    # TODO: Replace this stub with actual cricket model inference.
    return CricketDeliveryPrediction(
        match="dummy-cricket-stub",
        matchid=-1,
        inningsid=-1,
        overid=-1,
        ball_in_over=-1,
        ballid=-1,
        xlsx_overs="stub",
        scorecard_overs="stub",
        kph=-1.0,
        release_y=-999.0,
        release_z=-999.0,
        bounce_x=-999.0,
        bounce_y=-999.0,
        impact_x=-999.0,
        impact_y=-999.0,
        impact_z=-999.0,
        interception_distance=-999.0,
        stump_y=-999.0,
        stump_z=-999.0,
        swing_angle=-999.0,
        deviation=-999.0,
        runs=-1,
        wickets=-1,
    )
