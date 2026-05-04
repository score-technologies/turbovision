import time
from fastapi import HTTPException
from scorevision.miner.private_track.predictor import (
    is_cricket_request,
    predict_actions,
    predict_cricket_delivery,
)
from scorevision.miner.private_track.video import delete_video, download_video
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse
from scorevision.miner.private_track.logging import logger


async def handle_challenge(request: ChallengeRequest) -> ChallengeResponse:
    logger.info(f"Challenge received: {request.challenge_id}")
    start_time = time.perf_counter()
    video_path = None

    try:
        if is_cricket_request(request):
            prediction = predict_cricket_delivery(request)
            processing_time = time.perf_counter() - start_time

            logger.info(
                f"Cricket challenge completed: {request.challenge_id}, prediction: stub, time: {processing_time:.1f}s"
            )

            return ChallengeResponse(
                challenge_id=request.challenge_id,
                prediction=prediction,
                processing_time=processing_time,
            )

        video_path = await download_video(request.video_url)
        predictions = predict_actions(video_path)
        processing_time = time.perf_counter() - start_time

        logger.info(f"Challenge completed: {request.challenge_id}, predictions: {len(predictions)}, time: {processing_time:.1f}s")

        return ChallengeResponse(
            challenge_id=request.challenge_id,
            predictions=predictions,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Challenge failed: {request.challenge_id}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if video_path:
            delete_video(video_path)

