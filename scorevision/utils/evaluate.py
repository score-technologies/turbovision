from logging import getLogger

from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
    parse_challenge_type,
)
from scorevision.utils.data_models import (
    SVChallenge,
    SVRunOutput,
    SVEvaluation,
)
from scorevision.chute_template.schemas import TVPredictInput

from scorevision.vlm_pipeline.non_vlm_scoring.keypoints import evaluate_keypoints
from scorevision.utils.manifest import Manifest, ElementPrefix, PillarName

from scorevision.utils.settings import get_settings
from scorevision.utils.video_processing import FrameStore
from scorevision.vlm_pipeline.utils.data_models import (
    PseudoGroundTruth,
    MinerScore,
    AggregatedScore,
)
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    BoundingBox,
    ShirtColor,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
    OBJECT_ID_LOOKUP,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import Action
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import bbox_smoothness_per_type
from scorevision.utils.pillar_metric_registry import METRIC_REGISTRY

# NOTE: The following imports are required to load METRIC_REGISTRY
import scorevision.vlm_pipeline.non_vlm_scoring.keypoints
import scorevision.vlm_pipeline.non_vlm_scoring.objects
import scorevision.vlm_pipeline.non_vlm_scoring.smoothness

logger = getLogger(__name__)


def parse_miner_prediction(miner_run: SVRunOutput) -> dict[int, dict]:
    predicted_frames = (
        (miner_run.predictions or {}).get("frames") if miner_run.predictions else None
    ) or []
    logger.info(f"Miner predicted {len(predicted_frames)} frames")

    miner_annotations = {}
    for predicted_frame in predicted_frames:
        bboxes = []
        frame_number = predicted_frame.get("frame_id", -1)
        for bbox in predicted_frame.get("boxes", []) or []:
            try:
                raw_cls = bbox.get("cls_id")
                try:
                    object_id = int(raw_cls)
                except (TypeError, ValueError):
                    object_id = None

                looked_up = (
                    OBJECT_ID_LOOKUP.get(object_id) if object_id is not None else None
                )

                object_type: ObjectOfInterest
                object_colour: ShirtColor = ShirtColor.OTHER

                if looked_up is None:
                    object_type = ObjectOfInterest.PLAYER

                elif isinstance(looked_up, str):
                    team_str = looked_up.strip().lower().replace(" ", "")
                    object_type = ObjectOfInterest.PLAYER
                    if team_str == "team1":
                        object_colour = TEAM1_SHIRT_COLOUR
                    elif team_str == "team2":
                        object_colour = TEAM2_SHIRT_COLOUR
                    else:
                        object_colour = ShirtColor.OTHER

                else:
                    object_type = looked_up
                    team_field = (
                        (bbox.get("team") or bbox.get("team_id") or "").strip().lower()
                    )
                    if team_field in {"1", "team1"}:
                        object_colour = TEAM1_SHIRT_COLOUR
                    elif team_field in {"2", "team2"}:
                        object_colour = TEAM2_SHIRT_COLOUR
                    else:
                        object_colour = ShirtColor.OTHER

                bboxes.append(
                    BoundingBox(
                        bbox_2d=[
                            int(bbox["x1"]),
                            int(bbox["y1"]),
                            int(bbox["x2"]),
                            int(bbox["y2"]),
                        ],
                        label=object_type,
                        cluster_id=object_colour,
                    )
                )
            except Exception as e:
                logger.error(e)
                continue
        miner_annotations[frame_number] = {
            "bboxes": bboxes,
            "action": predicted_frame.get("action", None),
            "keypoints": predicted_frame.get("keypoints", []),
        }
    return miner_annotations


def post_vlm_ranking(
    payload: TVPredictInput,
    miner_run: SVRunOutput,
    challenge: SVChallenge,
    pseudo_gt_annotations: list[PseudoGroundTruth],
    frame_store: FrameStore,
    manifest: Manifest,
) -> SVEvaluation:
    settings = get_settings()
    logger.info(payload.meta)

    challenge_type = challenge.challenge_type
    if challenge_type is None:
        challenge_type = parse_challenge_type(payload.meta.get("challenge_type"))

    predicted_frames = (
        ((miner_run.predictions or {}).get("frames") or [])
        if miner_run.predictions
        else []
    )
    frame_count = len(predicted_frames)

    breakdown_dict = {}

    if (
        miner_run.success
        and frame_count >= settings.SCOREVISION_VIDEO_MIN_FRAME_NUMBER
        and frame_count <= settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER
        and challenge_type is not None
    ):
        breakdown_dict = get_element_scores(
            manifest=manifest,
            pseudo_gt_annotations=pseudo_gt_annotations,
            miner_run=miner_run,
            frame_store=frame_store,
            challenge_type=challenge_type,
        )
    else:
        logger.info(
            f"Miner success={miner_run.success} frames={frame_count} "
            f"challenge_type={getattr(challenge_type, 'value', None)} (must not be None)."
        )
    details = {
        "breakdown": breakdown_dict,
        # "group_scores": {
        #    "objects": objects_score,
        #    "keypoints": keypoints_score,
        # },
        "challenge": {
            "id_hash": challenge.challenge_id,
            "api_task_id": challenge.api_task_id,
            "type": getattr(challenge.challenge_type, "value", None),
        },
        "prompt": challenge.prompt,
    }
    logger.info(details)

    final_score = breakdown_dict.get("mean_weighted", 0.0)
    return SVEvaluation(
        acc_breakdown=breakdown_dict,
        latency_ms=miner_run.latency_ms,
        acc=final_score,
        score=final_score,
        details=details,
    )


def get_element_scores(
    manifest: Manifest,
    pseudo_gt_annotations: list[PseudoGroundTruth],
    miner_run: SVRunOutput,
    frame_store: FrameStore,
    challenge_type: ChallengeType,
) -> dict:
    settings = get_settings()
    miner_annotations = parse_miner_prediction(miner_run=miner_run)
    element_scores = {}
    for element in manifest.elements:
        pillar_scores = {}
        for pillar, weight in element.metrics.pillars.items():
            metric_fn = METRIC_REGISTRY.get((element.category, pillar))
            if metric_fn is None:
                raise NotImplementedError(
                    f"Could not compute score for pillar {pillar} in element of type {element.category}: A metric has yet to be defined and/or registered with @register_metric"
                )
            score = metric_fn(
                pseudo_gt=pseudo_gt_annotations,
                miner_predictions=miner_annotations,
                video_bboxes=[
                    miner_annotations[frame_num]["bboxes"]
                    for frame_num in sorted(miner_annotations.keys())
                ],
                image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                image_width=settings.SCOREVISION_IMAGE_WIDTH,
                frames=frame_store,
                challenge_type=challenge_type,
            )
            pillar_scores[pillar] = dict(score=score, weighted_score=score * weight)
        pillar_scores.update(
            dict(
                total_raw=sum(score["score"] for score in pillar_scores.values()),
                total_weighted=sum(
                    score["weighted_score"] for score in pillar_scores.values()
                ),
            )
        )
        element_scores[element.category] = pillar_scores
    element_score_values = list(element_scores.values())
    total_raw = sum(score["total_raw"] for score in element_score_values)
    total_weighted = sum(score["total_weighted"] for score in element_score_values)
    n_elements = len(manifest.elements)
    logger.info(f"Dividing score equally among {n_elements} Elements")
    weighted_mean = total_weighted / n_elements if n_elements else 0.0
    element_scores.update(
        dict(
            total_raw=total_raw,
            total_weighted=total_weighted,
            mean_weighted=weighted_mean,
        )
    )
    logger.info(element_scores)
    return element_scores
