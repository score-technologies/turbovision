from logging import getLogger
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Generator

from numpy import zeros

from scorevision.soccernet_utils import *
from scorevision.utils.settings import get_settings
from scorevision.utils.miner_registry import get_miners_from_registry, Miner
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.video_processing import FrameStore
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)
from scorevision.chute_template.schemas import TVPredictInput
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.utils.data_models import SVChallenge
from scorevision.vlm_pipeline.utils.response_models import FrameAnnotation, BoundingBox
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    FOOTBALL_DEFAULT_CATEGORY,
    FOOTBALL_CATEGORY_CONFIDENCE,
    FOOTBALL_REASON_PREFIX,
    OBJECT_ID_LOOKUP,
    Person,
)

logger = getLogger(__name__)
GT_DATASET_PATH = Path("benchmark_data")
ANNOTATIONS_PATH = GT_DATASET_PATH / "annotations"
ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)
CACHED_VIDEO_PATH = GT_DATASET_PATH / "videos"
CACHED_VIDEO_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class GroundTruth:
    challenge_type: ChallengeType
    url: str
    annotations: list[PseudoGroundTruth]
    name: str


def load_ground_truth_dataset(
    dataset_name: str = "gamestate-2025", data_split: str = "test"
) -> Generator[GroundTruth, None, None]:
    dataset_directory = GT_DATASET_PATH / dataset_name
    if not dataset_directory.is_dir() or not any(dataset_directory.iterdir()):
        logger.info("Dataset not found. Downloading from soccernet")
        download_raw_dataset(
            save_directory=str(GT_DATASET_PATH),
            dataset_name=dataset_name,
            dataset_split=data_split,
        )
    path_zipfile = dataset_directory / f"{data_split}.zip"
    if path_zipfile.exists() and not path_zipfile.with_suffix("").is_dir():
        logger.info("Unizipping dataset")
        unzip_raw_dataset(path_zipfile=path_zipfile)

    dataset = load_annotations_for_videos(
        dataset_directory=dataset_directory / data_split,
        annotation_json_filename="Labels-GameState",
        image_subdirectory_name="img1",
        video_path=CACHED_VIDEO_PATH,
        annotations_path=ANNOTATIONS_PATH,
    )

    logger.info("Dataset loaded")
    challenge_type = ChallengeType.FOOTBALL
    gts = []
    for videoname, ground_truth in dataset:
        logger.info(f"Formatting {videoname} as PseudoGroundTruth")
        gt = GroundTruth(
            name=videoname,
            challenge_type=challenge_type,
            url=f"https://scoredata.me/benchmark/{challenge_type.value}/{videoname}.mp4",
            annotations=[],
        )
        # frame_store = FrameStore(CACHED_VIDEO_PATH / f"{videoname}.mp4")
        annotations = [
            PseudoGroundTruth(
                video_name=videoname,
                frame_number=data.frame_id,
                spatial_image=zeros(
                    (2, 2)
                ),  # NOTE: this isnt used in post-vlm-metrics #gt.frame_store.get_frame(data.frame_id - 1),
                temporal_image=zeros(
                    (2, 2)
                ),  # NOTE: this isnt used in post-vlm-metrics #gt.frame_store.get_flow(data.frame_id - 1),
                annotation=FrameAnnotation(
                    bboxes=[
                        BoundingBox(
                            bbox_2d=(
                                bbox.x1,
                                bbox.y1,
                                bbox.x2,
                                bbox.y2,
                            ),
                            label=(
                                Person.PLAYER
                                if 6 <= bbox.cls_id <= 7
                                else OBJECT_ID_LOOKUP[bbox.cls_id]
                            ),
                            cluster_id=ID_TO_SHIRT_COLOUR.get(
                                bbox.cls_id, ShirtColor.OTHER
                            ),
                        )
                        for bbox in data.boxes
                    ],
                    category=FOOTBALL_DEFAULT_CATEGORY,
                    confidence=FOOTBALL_CATEGORY_CONFIDENCE,
                    reason=f"{FOOTBALL_REASON_PREFIX} players/referees/goalkeeper via palette + ball if present.",
                ),
            )
            for data in ground_truth[1:]
        ]
        gt.annotations = annotations
        yield gt


async def get_winning_miner() -> Miner | None:
    settings = get_settings()
    # miners = await get_miners_from_registry(netuid=settings.SCOREVISION_NETUID)

    # TODO: Mikhael replace this:

    from unittest.mock import MagicMock

    miner = MagicMock("Miner")
    miner.slug = "score-test-turbovision-mterryjack-tarpon"
    miner.chute_id = "0b87cec7-9faa-5248-b0a7-a0234c7e9363"
    return miner


async def run_benchmark_on_best_miner() -> None:
    logger.info("Loading GT dataset")
    gt_dataset = load_ground_truth_dataset()

    logger.info("Fetching winning miner")
    miner = await get_winning_miner()
    if miner is None:
        logger.info("No winning miner found")
        return

    logger.info("Evaluating miner on GT dataset")
    for gt in gt_dataset:
        frame_store = FrameStore(CACHED_VIDEO_PATH / f"{gt.name}.mp4")
        payload = TVPredictInput(url=gt.url, meta={})
        challenge = SVChallenge(
            env="SVEnv",
            payload=payload,
            meta={},
            prompt="ScoreVision benchmarking",
            challenge_id=gt.challenge_type.value,
            frame_numbers=list(range(750)),
            frames=[],
            dense_optical_flow_frames=[],
            challenge_type=gt.challenge_type,
        )

        logger.info(f"Calling model {miner.slug}")
        miner_run = await call_miner_model_on_chutes(
            slug=miner.slug,
            chute_id=miner.chute_id,
            payload=payload,
        )

        logger.info("post VLM evaluation")
        evaluation = post_vlm_ranking(
            payload=payload,
            miner_run=miner_run,
            challenge=challenge,
            pseudo_gt_annotations=gt.annotations,
            frame_store=frame_store,
        )
        results = asdict(evaluation)

        logger.info(f"saving results to R2: {results}")
    #     #TODO: add note when saving that this is benchmark data
    #     await emit_shard(
    #         slug=miner.slug,
    #         challenge=challenge,
    #         miner_run=miner_run,
    #         evaluation=evaluation,
    #         miner_hotkey_ss58=miner.hotkey,
    #     )


if __name__ == "__main__":
    from asyncio import run
    from logging import basicConfig, INFO

    basicConfig(level=INFO)
    run(run_benchmark_on_best_miner())
