from logging import getLogger

from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
    parse_challenge_type,
)
from scorevision.utils.data_models import (
    SVChallenge,
    SVRunOutput,
    SVEvaluation,
    # TotalScore,
)
from scorevision.chute_template.schemas import TVPredictInput

from scorevision.vlm_pipeline.non_vlm_scoring.keypoints import evaluate_keypoints
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    compare_object_counts,
    compare_team_labels,
    compare_object_labels,
    compare_object_placement,
)
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
    miner_annotations = parse_miner_prediction(miner_run=miner_run)
    logger.info(payload.meta)

    challenge_type = challenge.challenge_type
    if challenge_type is None:
        challenge_type = parse_challenge_type(payload.meta.get("challenge_type"))

    if (
        miner_run.success
        and len(miner_annotations) == settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER
        and challenge_type is not None
    ):
        total, breakdown_dict = get_element_scores(
            manifest=manifest,
            pseudo_gt_annotations=pseudo_gt_annotations,
            miner_annotations=miner_annotations,
            frame_store=frame_store,
        )

    else:
        final_score, breakdown_dict = 0.0, {}
        logger.info(
            f"Miner success={miner_run.success} frames={len(miner_annotations)} "
            f"challenge_type={getattr(challenge_type, 'value', None)} (must not be None)."
        )

    details = {
        "breakdown": breakdown_dict,
        "group_scores": {
            "objects": objects_score,
            "keypoints": keypoints_score,
        },
        "challenge": {
            "id_hash": challenge.challenge_id,
            "api_task_id": challenge.api_task_id,
            "type": getattr(challenge.challenge_type, "value", None),
        },
        "prompt": challenge.prompt,
    }
    logger.info(details)

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
    miner_annotations: dict[int, dict],
    frame_store: FrameStore,
) -> tuple[float, dict]:
    METRICS: dict[ElementPrefix, dict[PillarName, callable]] = {
        ElementPrefix.PLAYER_DETECTION: {
            PillarName.IOU: lambda: compare_object_placement(
                pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
            ),
            PillarName.COUNT: lambda: compare_object_counts(
                pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
            ),
            PillarName.SMOOTHNESS: lambda: bbox_smoothness_per_type(
                video_bboxes=[
                    miner_annotations[frame_num]["bboxes"]
                    for frame_num in sorted(miner_annotations.keys())
                ],
                image_height=settings.SCOREVISION_IMAGE_HEIGHT,
                image_width=settings.SCOREVISION_IMAGE_WIDTH,
            ),
            PillarName.ROLE: lambda: compare_object_labels(
                pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
            )
            + compare_team_labels(
                pseudo_gt=pseudo_gt_annotations, miner_predictions=miner_annotations
            ),
        },
        ElementPrefix.BALL_DETECTION: {},
        ElementPrefix.PITCH_CALIBRATION: {
            PillarName.IOU: lambda: evaluate_keypoints(
                frames=frame_store,
                miner_predictions=miner_annotations,
                challenge_type=challenge_type,
            )
        },
    }

    element_scores = {}
    for element in manifest.elements:
        pillar_scores = {}
        for pillar, weight in element.metrics.items():
            metric = METRICS[element.category].get(pillar)
            if metric is None:
                raise NotImplemented(
                    f"No {pillar} metric defined for {element.category}"
                )
            score = metric()
            pillar_scores[pillar] = dict(raw=score, weighted=score * weight)
        pillar_scores.update(
            dict(
                total_raw=sum(score for score["raw"] in pillar_scores),
                total_weighted=sum(score for score["weighted"] in pillar_scores),
            )
        )
        element_scores[element.category] = pillar_scores
    total_raw = sum(score for score["total_raw"] in element_scores)
    total_weighted = sum(score for score["total_weighted"] in element_scores)
    element_scores.update(
        dict(
            total_raw=total_raw,
            total_weighted=total_weighted,
        )
    )
    logger.info(element_scores)
    n_elements = len(manifest.elements)
    logger.info(f"Dividing score equally among {n_elements} Elements")
    weighted_mean = sum(
        score["total_weighted"] / n_elements for score in element_scores
    )
    logger.info(f"Weighted Mean: {weighted_mean}")
    return weighted_mean, element_scores
