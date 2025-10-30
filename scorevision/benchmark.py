from logging import getLogger
from unittest.mock import MagicMock
from pathlib import Path
from dataclasses import dataclass

from SoccerNet.Downloader import SoccerNetDownloader

from scorevision.utils.settings import get_settings
from scorevision.utils.miner_registry import get_miners_from_registry, Miner
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.video_processing import FrameStore
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
    CHALLENGE_ID_LOOKUP,
)
from scorevision.chute_template.schemas import TVPredictInput
from scorevision.utils.cloudflare_helpers import emit_shard
from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.utils.data_models import SVChallenge
from scorevision.vlm_pipeline.utils.response_models import FrameAnnotation, BoundingBox

logger = getLogger(__name__)
CACHED_VIDEO_PATH = Path("cached_gt_videos")


@dataclass
class GroundTruth:
    challenge_type: ChallengeType
    url: str
    annotations: list[PseudoGroundTruth]

    @property
    def challenge_id(self) -> int:
        return CHALLENGE_ID_LOOKUP[self.challenge_type]

    # @property
    # def frame_store(self) -> FrameStore:
    #    if self.frame_store is None:
    #        self.frame_store = FrameStore(CACHED_VIDEO_PATH / self.url.stem)
    #    return self.frame_store


def load_ground_truth_dataset(dataset_directory: Path) -> list[GroundTruth]:
    def download_raw_dataset(
        save_directory: str, dataset_name: str, dataset_split: str
    ) -> None:
        settings = get_settings()
        downloader = SoccerNetDownloader(LocalDirectory=save_directory)
        if settings.SOCCERNET_TOKEN.get_secret_value() is None:
            raise Exception("SOCCERNET_TOKEN not set")
        downloader.password = settings.SOCCERNET_TOKEN.get_secret_value()
        downloader.downloadDataTask(task=dataset_name, split=[dataset_split])
        logger.info(f"Dataset downloaded to {save_directory}")

    if not dataset_directory.is_dir() or not any(dataset_directory.iterdir()):
        logger.info("Dataset not found. Downloading from soccernet")
        download_raw_dataset(
            save_directory=str(dataset_directory.parent),
            dataset_name="gamestate-2025",
            dataset_split="test",
        )

    # gts =  []
    # for _ in range(10):
    #     challenge_type = ChallengeType.SOCCER
    #     gt = GroundTruth(
    #         challenge_type = challenge_type,
    #         url = f"https://scoredata.me/benchmark/{challenge_type.value}/{videoname}.mp4",
    #         annotations = []
    #     )
    #     videoname = "" #TODO
    #     frame_id = 1 #TODO
    #     annotations = [
    #         PseudoGroundTruth(
    #             video_name=videoname,
    #             frame_number=frame_id,
    #             spatial_image=gt.frame_store.get_frame(frame_id - 1),
    #             temporal_image=gt.frame_store.get_flow(frame_id - 1),
    #             annotation=FrameAnnotation(
    #                 bboxes=[
    #                     BoundingBox(
    #                         bbox_2d=(
    #                             bbox["x1"],
    #                             bbox["y1"],
    #                             bbox["x2"],
    #                             bbox["y2"],
    #                         ),
    #                         label=(
    #                             Person.PLAYER
    #                             if 6 <= int(bbox["cls_id"]) <= 7
    #                             else OBJECT_ID_LOOKUP[int(bbox["cls_id"])]
    #                         ),
    #                         cluster_id=ID_TO_SHIRT_COLOUR.get(
    #                             int(bbox["cls_id"]), ShirtColor.OTHER
    #                         ),
    #                     )
    #                     for bbox in data["boxes"]
    #                 ],
    #                 category=FOOTBALL_DEFAULT_CATEGORY,
    #                 confidence=FOOTBALL_CATEGORY_CONFIDENCE,
    #                 reason=f"{FOOTBALL_REASON_PREFIX} players/referees/goalkeeper via palette + ball if present.",
    #             ),
    #         )
    #         for data in ground_truth[1:]
    #     ]
    #     gt.annotations = annotations
    #     gts.append(gt)
    # return gts


def get_winning_miner() -> Miner:
    miners = get_miners_from_registry()
    return miners[0]
    # TODO: Mikhael


async def run_benchmark_on_best_miner() -> None:
    logger.info("Loading GT dataset")
    gt_dataset = load_ground_truth_dataset(
        dataset_directory=Path("benchmark_data/gamestate-2025")
    )

    # logger.info("Fetching winning miner")
    # miner = get_winning_miner()

    # logger.info("Evaluating miner on GT dataset")
    # for gt in gt_dataset:
    #     payload = TVPredictInput(
    #         url=gt.url
    #         meta={"challenge_type": gt.challenge_id}
    #     )
    #     challenge = SVChallenge(
    #         env="SVEnv",
    #         payload=payload,
    #         meta={},
    #         prompt="ScoreVision benchmarking",
    #         challenge_id=gt.challenge_id,
    #         frame_numbers=list(range(750)),
    #         frames=[],
    #         dense_optical_flow_frames=[],
    #         challenge_type=gt.challenge_type,
    #     )

    #     logger.info(f"Calling model {miner.slug}")
    #     miner_run = await call_miner_model_on_chutes(
    #         slug=miner.slug,
    #         chute_id=miner.chute_id,
    #         payload=payload,
    #     )

    #     logger.info("post VLM evaluation")
    #     evaluation = post_vlm_ranking(
    #         payload=payload,
    #         miner_run=miner_run,
    #         challenge=challenge,
    #         pseudo_gt_annotations=gt.annotations,
    #         frame_store=gt.frame_store,
    #     )
    #     results = asdict(evaluation)

    #     logger.info("saving results to R2")
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

    run(run_benchmark_on_best_miner())
