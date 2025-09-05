from logging import getLogger
from time import monotonic

from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.vlm_pipeline.vlm_annotator import (
    generate_annotations_for_select_frames,
)
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.data_models import SVChallenge, SVPredictResult, SVRunOutput
from scorevision.utils.chutes_helpers import get_chute_slug_and_id
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.settings import get_settings
from scorevision.utils.evaluate import evaluate_using_vlms, post_vlm_ranking


logger = getLogger(__name__)


async def vlm_pipeline(hf_revision: str, local_model: bool) -> None:
    """Run a single miner on the VLM pipeline off-chain"""
    settings = get_settings()
    challenge_data = {
        "task_id": "mock-challenge",
        "video_url": "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4",
    }
    logger.info(f"Challenge data from API: {challenge_data}")
    payload, frame_numbers, frames, flows = await prepare_challenge_payload(
        challenge=challenge_data
    )
    if not payload:
        raise Exception("Failed to prepare payload from challenge.")

    payload.frames = payload.frames[: settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER]
    if local_model:
        logger.info("Calling model from mock chutes API")
        base_url = "http://localhost:8000"
        session = await get_async_client()
        t0 = monotonic()
        async with session.post(
            f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
            },
            json=payload.model_dump(mode="json"),
        ) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
            if response.status != 200:
                raise Exception("Non-200 response from predict")
            output = await response.json()
            res = SVPredictResult(
                success=bool(output.get("success", True)),
                model=output.get("model"),
                latency_seconds=monotonic() - t0,
                predictions=output.get("predictions") or output.get("data"),
                error=output.get("error"),
                raw=output,
            )
            miner_output = SVRunOutput(
                success=res.success,
                latency_ms=res.latency_seconds * 1000.0,
                predictions=res.predictions if res.success else None,
                error=res.error,
                model=res.model,
            )
    else:
        chute_slug, _ = await get_chute_slug_and_id(revision=hf_revision)
        logger.info("Calling model from chutes API")
        miner_output = await call_miner_model_on_chutes(
            slug=chute_slug,
            payload=payload,
        )
    logger.info(f"Miner: {miner_output}")

    challenge = SVChallenge(
        env="SVEnv",
        payload=payload,
        meta={},
        prompt="ScoreVision video task mock-challenge",
        challenge_id="mock-challenge",
        frame_numbers=frame_numbers,
        frames=frames,
        dense_optical_flow_frames=flows,
    )
    # logger.info(f"Challenge: {challenge}")
    pseudo_gt_annotations = await generate_annotations_for_select_frames(
        video_name=challenge.challenge_id,
        frames=challenge.frames,
        flow_frames=challenge.dense_optical_flow_frames,
        frame_numbers=challenge.frame_numbers,
    )
    logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")

    vlm_evaluation = await evaluate_using_vlms(
        challenge=challenge,
        miner_run=miner_output,
        pseudo_gt_annotations=pseudo_gt_annotations,
    )
    logger.info(f"VLM Evaluation: {vlm_evaluation}")

    evaluation = post_vlm_ranking(
        miner_run=miner_output,
        challenge=challenge,
        miner_score=vlm_evaluation,
    )
    logger.info(f"Evaluation: {evaluation}")
