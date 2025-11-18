from logging import getLogger
from time import monotonic
from pathlib import Path

from huggingface_hub import HfApi

from scorevision.utils.manifest import Manifest
from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.vlm_pipeline.vlm_annotator import (
    generate_annotations_for_select_frames,
)
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.data_models import SVChallenge, SVPredictResult, SVRunOutput
from scorevision.utils.chutes_helpers import (
    get_chute_slug_and_id,
)
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import (
    filter_low_quality_pseudo_gt_annotations,
)
from scorevision.utils.data_models import SVEvaluation
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)
from scorevision.utils.huggingface_helpers import get_huggingface_repo_revision
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


async def run_vlm_pipeline_once_for_single_miner(
    hf_revision: str | None,
) -> SVEvaluation:
    """Run a single miner on the VLM pipeline off-chain
    NOTE: This flow should match the flow in the runner"""
    manifest = Manifest.load_yaml(path=Path("example_manifest.yml"))
    logger.info(f"Manifest loaded: {manifest}")
    challenge_data = {
        "task_id": "0",
        "video_url": "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4",
    }
    logger.info(f"Challenge data from API: {challenge_data}")
    if not hf_revision:
        settings = get_settings()
        hf_api = HfApi(token=settings.HUGGINGFACE_API_KEY.get_secret_value())
        hf_revision = await get_huggingface_repo_revision(hf_api=hf_api)

    payload, frame_numbers, frames, flows, frame_store = (
        await prepare_challenge_payload(challenge=challenge_data)
    )
    if not payload:
        raise Exception("Failed to prepare payload from challenge.")

    chute_slug, chute_id = await get_chute_slug_and_id(revision=hf_revision)
    if not chute_slug:
        raise Exception("Failed to fetch chute slug")

    logger.info("Calling model from chutes API")
    miner_output = await call_miner_model_on_chutes(
        slug=chute_slug,
        chute_id=chute_id,
        payload=payload,
    )
    logger.info(f"Miner: {miner_output}")

    challenge = SVChallenge(
        env="SVEnv",
        payload=payload,
        meta={},
        prompt="ScoreVision video task mock-challenge",
        challenge_id="0",
        frame_numbers=frame_numbers,
        frames=frames,
        dense_optical_flow_frames=flows,
        challenge_type=ChallengeType.FOOTBALL,
    )
    # logger.info(f"Challenge: {challenge}")
    pseudo_gt_annotations = await generate_annotations_for_select_frames(
        video_name=challenge.challenge_id,
        frames=challenge.frames,
        flow_frames=challenge.dense_optical_flow_frames,
        frame_numbers=challenge.frame_numbers,
    )
    logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")
    pseudo_gt_annotations = filter_low_quality_pseudo_gt_annotations(
        annotations=pseudo_gt_annotations
    )
    logger.info(
        f"{len(pseudo_gt_annotations)} Pseudo GT annotations had sufficient quality"
    )

    evaluation = post_vlm_ranking(
        payload=payload,
        miner_run=miner_output,
        challenge=challenge,
        pseudo_gt_annotations=pseudo_gt_annotations,
        frame_store=frame_store,
        manifest=manifest,
    )
    frame_store.unlink()
    logger.info(f"Evaluation: {evaluation}")
    return evaluation
