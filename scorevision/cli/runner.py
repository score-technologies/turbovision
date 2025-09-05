from logging import getLogger
import os
import random
import asyncio

from scorevision.utils.settings import get_settings
from scorevision.utils.challenges import get_challenge_from_scorevision
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.evaluate import evaluate_using_vlms, post_vlm_ranking
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.utils.async_clients import close_http_clients
from scorevision.vlm_pipeline.vlm_annotator import (
    generate_annotations_for_select_frames,
)
from scorevision.utils.miner_registry import get_miners_from_registry, Miner
from scorevision.utils.bittensor_helpers import get_subtensor

logger = getLogger(__name__)


async def runner(slug: str | None = None) -> None:
    settings = get_settings()
    NETUID = settings.SCOREVISION_NETUID
    MAX_MINERS = int(os.getenv("SV_MAX_MINERS_PER_RUN", "60"))

    try:
        miners = await get_miners_from_registry(NETUID)
        if not miners:
            logger.warning("No eligible miners found on-chain.")
            return
        challenge, payload = await get_challenge_from_scorevision()
        logger.info(f"Challenge: {challenge}")

        miner_list = list(miners.values())
        random.shuffle(miner_list)
        miner_list = miner_list[: min(MAX_MINERS, len(miner_list))]

        pseudo_gt_annotations = await generate_annotations_for_select_frames(
            video_name=challenge.challenge_id,
            frames=challenge.frames,
            flow_frames=challenge.dense_optical_flow_frames,
            frame_numbers=challenge.frame_numbers,
        )
        logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")
        for m in miner_list:
            try:
                miner_output = await call_miner_model_on_chutes(
                    slug=m.slug,
                    payload=payload,
                )
                logger.info(f"Miner: {miner_output}")

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

                await emit_shard(
                    slug=m.slug,
                    challenge=challenge,
                    miner_run=miner_output,
                    evaluation=evaluation,
                    miner_hotkey_ss58=m.hotkey,
                )
            except Exception as e:
                logger.warning(
                    "Miner uid=%s slug=%s failed: %s",
                    getattr(m, "uid", "?"),
                    getattr(m, "slug", "?"),
                    e,
                )
                continue
    except Exception as e:
        logger.error(e)
    finally:
        close_http_clients()


async def runner_loop():
    """Runs `runner()` every N blocks (default: 300)."""
    settings = get_settings()
    TEMPO = 300

    st = None
    last_block = -1

    while True:
        try:
            if st is None:
                st = await get_subtensor()

            block = await st.get_current_block()

            if block <= last_block or block % TEMPO != 0:
                await st.wait_for_block()
                continue

            logger.info(f"[RunnerLoop] Triggering runner at block {block}")
            await runner()

            last_block = block

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[RunnerLoop] Error: {e}; retrying…")
            st = None
            await asyncio.sleep(120)
