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
from scorevision.utils.manifest import Manifest

from scorevision.utils.settings import get_settings
from scorevision.utils.video_processing import FrameStore
from scorevision.vlm_pipeline.utils.data_models import (
    PseudoGroundTruth,
)
from scorevision.vlm_pipeline.utils.response_models import (
    BoundingBox,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.utils.pillar_metric_registry import (
    METRIC_REGISTRY,
)

# NOTE: The following imports are required to load METRIC_REGISTRY
import scorevision.vlm_pipeline.non_vlm_scoring.keypoints
import scorevision.vlm_pipeline.non_vlm_scoring.objects
import scorevision.vlm_pipeline.non_vlm_scoring.smoothness
from scorevision.utils.rtf import (
    calculate_rtf,
    check_rtf_gate,
    get_service_rate_fps_for_element,
)

logger = getLogger(__name__)


def parse_miner_prediction(
    miner_run: SVRunOutput, object_names: list[str]
) -> dict[int, dict]:
    predicted_frames = (
        (miner_run.predictions or {}).get("frames") if miner_run.predictions else None
    ) or []
    logger.info(f"Miner predicted {len(predicted_frames)} frames")

    miner_annotations = {}
    for predicted_frame in predicted_frames:
        bboxes = []
        frame_number = predicted_frame.get("frame_id", -1)
        if any(object_names):
            for bbox in predicted_frame.get("boxes", []) or []:
                try:
                    raw_cls = bbox.get("cls_id")
                    try:
                        object_id = int(raw_cls)
                    except (TypeError, ValueError):
                        object_id = None

                    looked_up = None
                    if object_id is not None and 0 <= object_id < len(object_names):
                        looked_up = object_names[object_id]
                    else:
                        continue

                    bboxes.append(
                        BoundingBox(
                            bbox_2d=[
                                int(bbox["x1"]),
                                int(bbox["y1"]),
                                int(bbox["x2"]),
                                int(bbox["y2"]),
                            ],
                            label=looked_up,
                            # cluster_id=object_colour,
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
    manifest: Manifest | None,
    element_id: str | None = None,
) -> SVEvaluation:
    settings = get_settings()
    logger.info(payload.meta)

    challenge_type = challenge.challenge_type
    if challenge_type is None:
        challenge_type = parse_challenge_type(payload.meta.get("challenge_type"))

    expected_total = int(payload.meta.get("n_frames_total") or 0)

    predicted_frames = (
        ((miner_run.predictions or {}).get("frames") or [])
        if miner_run.predictions
        else []
    )
    frame_count = len(predicted_frames)

    logger.info(
        "Frame check: miner_unique=%s expected_total=%s",
        frame_count,
        expected_total,
    )


    breakdown_dict = {}

    if (
        miner_run.success
        and frame_count >= settings.SCOREVISION_VIDEO_MIN_FRAME_NUMBER
        and frame_count <= expected_total
        and challenge_type is not None
        and manifest is not None
    ):
        breakdown_dict = get_element_scores(
            manifest=manifest,
            pseudo_gt_annotations=pseudo_gt_annotations,
            miner_run=miner_run,
            frame_store=frame_store,
            challenge_type=challenge_type,
            element_id=element_id,
        )
    else:
        logger.info(
            f"Miner success={miner_run.success} frames={frame_count} "
            f"challenge_type={getattr(challenge_type, 'value', None)} (must not be None)."
            f"manifest_present={manifest is not None}."
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

    p95_latency_ms = getattr(miner_run, "latency_p95_ms", None) or miner_run.latency_ms

    service_rate_fps = None
    if manifest is not None and element_id:
        service_rate_fps = get_service_rate_fps_for_element(
            manifest=manifest,
            element_id=element_id,
        )

    latency_pass = True
    rtf_value = None

    if service_rate_fps is None:
        logger.warning(
            "[RTF] service_rate_fps unavailable for element '%s'; "
            "skipping latency gate (config issue).",
            element_id,
        )
    else:
        try:
            rtf_value = calculate_rtf(
                p95_latency_ms=float(p95_latency_ms),
                service_rate_fps=float(service_rate_fps),
            )
            latency_pass = check_rtf_gate(rtf_value)
            logger.info(
                "[RTF] element_id=%s service_rate_fps=%.3f p95_ms=%.1f "
                "rtf=%.3f latency_pass=%s",
                element_id,
                float(service_rate_fps),
                float(p95_latency_ms),
                float(rtf_value),
                latency_pass,
            )
        except Exception as e:
            logger.warning(
                "[RTF] Error computing RTF for element '%s': %s. "
                "Skipping latency gate.",
                element_id,
                e,
            )
            latency_pass = True
            rtf_value = None

        if not latency_pass:
            logger.info(
                "[RTF] Failing latency gate (rtf=%.3f > 1.0) → forcing score=0.",
                rtf_value,
            )
            final_score = 0.0

    details.setdefault("latency", {})
    details["latency"].update(
        {
            "latency_ms": miner_run.latency_ms,
            "latency_p50_ms": getattr(miner_run, "latency_p50_ms", None),
            "latency_p95_ms": p95_latency_ms,
            "latency_p99_ms": getattr(miner_run, "latency_p99_ms", None),
            "latency_max_ms": getattr(miner_run, "latency_max_ms", None),
            "service_rate_fps": service_rate_fps,
            "rtf": rtf_value,
            "latency_pass": latency_pass,
        }
    )

    acc_value = final_score

    scored_frame_numbers = [pgt.frame_number for pgt in pseudo_gt_annotations]

    return SVEvaluation(
        acc_breakdown=breakdown_dict,
        latency_ms=miner_run.latency_ms,
        acc=acc_value,
        score=final_score,
        details=details,
        latency_p95_ms=p95_latency_ms,
        latency_pass=latency_pass,
        rtf=rtf_value,
        scored_frame_numbers=scored_frame_numbers,
    )


def get_element_scores(
    manifest: Manifest,
    pseudo_gt_annotations: list[PseudoGroundTruth],
    miner_run: SVRunOutput,
    frame_store: FrameStore,
    challenge_type: ChallengeType,
    element_id: str | None = None
) -> dict:
    settings = get_settings()

    elements = manifest.elements
    if element_id:
        e = manifest.get_element(id=element_id)
        if e is None:
            raise ValueError(f"Element {element_id} not found in manifest")
        elements = [e]

    element_scores = {}
    for element in elements:
        miner_annotations = parse_miner_prediction(
            miner_run=miner_run, object_names=element.objects or []
        )
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
                keypoints_template=element.keypoints,
            )
            pillar_scores[pillar] = dict(score=score, weighted_score=score * weight)

        total_raw = sum(score["score"] for score in pillar_scores.values())
        total_weighted = sum(
            score["weighted_score"] for score in pillar_scores.values()
        )
        pillar_scores.update(
            dict(
                total_raw=total_raw,
                total_weighted=total_weighted,
                total_weighted_and_gated=element.weight_score(score=total_weighted),
            )
        )
        element_scores[element.category] = pillar_scores
    element_score_values = list(element_scores.values())
    total_raw = sum(score["total_raw"] for score in element_score_values)
    total_weighted = sum(score["total_weighted"] for score in element_score_values)
    total_weighted_and_gated = sum(
        score["total_weighted_and_gated"] for score in element_score_values
    )
    n_elements = len(elements)
    logger.info(f"Dividing score equally among {n_elements} Elements")
    weighted_mean = total_weighted_and_gated / n_elements if n_elements else 0.0
    element_scores.update(
        dict(
            total_raw=total_raw,
            total_weighted=total_weighted,
            total_weighted_and_gated=total_weighted_and_gated,
            mean_weighted=weighted_mean,
        )
    )
    logger.info(element_scores)
    return element_scores
