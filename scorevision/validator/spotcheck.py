import asyncio
import logging
import random
from json import dumps
from pathlib import Path
from typing import Any
from scorevision.utils.challenges import build_svchallenge_from_parts, prepare_challenge_payload
from scorevision.utils.cloudflare_helpers import _verify_signature
from scorevision.utils.data_models import SVRunOutput
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.utils.manifest import Manifest, get_current_manifest, load_manifest_from_public_index
from scorevision.utils.r2_public import (
    extract_base_url,
    extract_block_from_key,
    fetch_index_keys,
    fetch_json_from_url,
    fetch_miner_predictions,
    fetch_responses_data,
    fetch_shard_lines,
    filter_keys_by_tail,
)
from scorevision.utils.settings import get_settings
from scorevision.validator.models import ChallengeRecord, SpotcheckResult
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import filter_low_quality_pseudo_gt_annotations
from scorevision.vlm_pipeline.vlm_annotator_sam3 import generate_annotations_for_select_frames_sam3

logger = logging.getLogger("scorevision.spotcheck")


def parse_challenge_record_from_line(line: dict, key: str) -> ChallengeRecord | None:
    try:
        payload_str = dumps(line.get("payload") or {}, sort_keys=True, separators=(",", ":"))
        sig = line.get("signature", "")
        validator_hk = line.get("hotkey", "")
        if validator_hk and sig and not _verify_signature(validator_hk, payload_str, sig):
            return None

        payload = line.get("payload") or {}

        miner_info = payload.get("miner") or {}
        miner_hotkey = (miner_info.get("hotkey") or "").strip()
        if not miner_hotkey:
            return None

        meta = payload.get("meta") or {}
        challenge_id = str(meta.get("task_id") or payload.get("task_id") or "").strip()
        if not challenge_id:
            return None

        element_id = str(payload.get("element_id") or "").strip()
        window_id = str(payload.get("window_id") or meta.get("window_id") or "").strip()

        evaluation = payload.get("evaluation") or {}
        central_score = float(evaluation.get("score", 0.0))

        run_info = payload.get("run") or {}
        responses_key = run_info.get("responses_key")

        scored_frame_numbers = payload.get("scored_frame_numbers")
        if scored_frame_numbers is not None and not isinstance(scored_frame_numbers, list):
            scored_frame_numbers = None

        block = extract_block_from_key(key) or 0

        return ChallengeRecord(
            challenge_id=challenge_id,
            element_id=element_id,
            window_id=window_id,
            block=block,
            miner_hotkey=miner_hotkey,
            central_score=central_score,
            payload=payload,
            miner_predictions=None,
            video_url=None,
            responses_key=responses_key,
            scored_frame_numbers=scored_frame_numbers,
        )
    except Exception as e:
        logger.warning("Error parsing shard line: %s", e)
        return None


async def fetch_random_challenge_record(
    tail_blocks: int,
    element_id: str | None = None,
) -> ChallengeRecord | None:
    settings = get_settings()
    public_url = settings.R2_BUCKET_PUBLIC_URL

    if not public_url:
        logger.error("R2_BUCKET_PUBLIC_URL not set - cannot fetch challenges")
        return None

    logger.info("Fetching random challenge from R2 (tail=%d blocks, element=%s)", tail_blocks, element_id)

    index_keys = await fetch_index_keys(public_url)
    if not index_keys:
        logger.warning("No keys found in index")
        return None

    filtered_keys, max_block, min_keep = filter_keys_by_tail(index_keys, tail_blocks)

    if not filtered_keys:
        logger.warning("No valid shard keys found within tail window")
        return None

    logger.info("Found %d shards within tail window (max_block=%d, min=%d)", len(filtered_keys), max_block, min_keep)

    random.shuffle(filtered_keys)
    candidates: list[ChallengeRecord] = []

    for key in filtered_keys[:20]:
        lines = await fetch_shard_lines(public_url, key)
        for line in lines:
            record = parse_challenge_record_from_line(line, key)
            if record is None:
                continue
            if element_id and record.element_id != element_id:
                continue
            candidates.append(record)
            if len(candidates) >= 50:
                break
        if len(candidates) >= 50:
            break

    if not candidates:
        logger.warning("No valid challenge records found in R2")
        return None

    random.shuffle(candidates)
    for record in candidates:
        if record.responses_key and public_url:
            predictions, video_url = await fetch_responses_data(record.responses_key, public_url)
            if predictions and video_url:
                record.miner_predictions = predictions
                record.video_url = video_url
                logger.info(
                    "Selected challenge: id=%s element=%s miner=%s central_score=%.4f",
                    record.challenge_id,
                    record.element_id,
                    record.miner_hotkey[:8] if record.miner_hotkey else "?",
                    record.central_score,
                )
                return record

    logger.warning("No challenges with available responses found")
    return None


def calculate_match_percentage(central_score: float, audit_score: float) -> float:
    if central_score == 0.0 and audit_score == 0.0:
        return 1.0
    if central_score == 0.0 or audit_score == 0.0:
        return 0.0
    diff = abs(central_score - audit_score)
    max_score = max(abs(central_score), abs(audit_score))
    return max(0.0, 1.0 - (diff / max_score))


def scores_match(central_score: float, audit_score: float, threshold: float) -> bool:
    return calculate_match_percentage(central_score, audit_score) >= threshold


def calculate_next_spotcheck_delay(min_interval_seconds: int, max_interval_seconds: int) -> float:
    return random.uniform(min_interval_seconds, max_interval_seconds)


async def load_manifest_for_spotcheck(block: int | None = None) -> Manifest | None:
    settings = get_settings()
    try:
        if getattr(settings, "URL_MANIFEST", None):
            cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
            return await load_manifest_from_public_index(
                settings.URL_MANIFEST,
                block_number=block,
                cache_dir=cache_dir,
            )
        return get_current_manifest(block_number=block)
    except Exception as e:
        logger.warning("Failed to load manifest: %s", e)
        return None


async def regenerate_ground_truth_sam3(
    challenge_record: ChallengeRecord,
    manifest: Manifest,
    required_n_frames: int = 3,
) -> tuple[Any, Any, Any, list] | None:
    if not challenge_record.video_url:
        logger.error("No video_url in challenge record %s", challenge_record.challenge_id)
        return None

    if not challenge_record.element_id:
        logger.error("No element_id in challenge record %s", challenge_record.challenge_id)
        return None

    element = manifest.get_element(id=challenge_record.element_id)
    challenge_type = element.keypoint_template.value if element and element.keypoint_template else "football"
    logger.info("Using challenge_type=%s for spotcheck", challenge_type)

    chal_api = {
        "task_id": challenge_record.challenge_id,
        "element_id": challenge_record.element_id,
        "video_url": challenge_record.video_url,
        "window_id": challenge_record.window_id,
        "challenge_type": challenge_type,
    }

    video_cache: dict[str, Any] = {}
    try:
        scored_frames = challenge_record.scored_frame_numbers
        if scored_frames:
            logger.info(
                "Using scored_frame_numbers from shard: %s",
                scored_frames,
            )
            effective_required = len(scored_frames)
        else:
            logger.info("No scored_frame_numbers in shard - using random frames")
            effective_required = required_n_frames

        payload, frame_numbers, frames, flows, frame_store = await prepare_challenge_payload(
            challenge=chal_api,
            video_cache=video_cache,
            frame_numbers=scored_frames,
        )

        logger.info("Prepared %d frames for SAM3 (required=%d)", len(frames), effective_required)

        if len(frames) < effective_required:
            logger.warning("Not enough frames (%d/%d) for SAM3", len(frames), effective_required)
            return None

        challenge = build_svchallenge_from_parts(
            chal_api=chal_api,
            payload=payload,
            frame_numbers=frame_numbers,
            frames=frames,
            flows=flows,
        )

        element = manifest.get_element(id=challenge_record.element_id)
        if element is None:
            logger.warning("Element %s not found in manifest", challenge_record.element_id)
            return None

        pseudo_gt_annotations = await generate_annotations_for_select_frames_sam3(
            video_name=challenge.challenge_id,
            frames=challenge.frames,
            flow_frames=challenge.dense_optical_flow_frames,
            frame_numbers=challenge.frame_numbers,
            element=element,
        )

        if not pseudo_gt_annotations:
            logger.warning("SAM3 generated no pseudo-GT annotations")
            return None

        filtered = filter_low_quality_pseudo_gt_annotations(annotations=pseudo_gt_annotations)
        logger.info(
            "Generated %d pseudo-GT annotations, %d after filtering",
            len(pseudo_gt_annotations),
            len(filtered),
        )

        if not filtered:
            logger.warning("All pseudo-GT annotations filtered out")
            return None

        return challenge, payload, frame_store, filtered

    except Exception as e:
        import traceback
        logger.error("Failed to regenerate ground truth: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        return None
    finally:
        cached_path = video_cache.get("path")
        if cached_path:
            try:
                Path(cached_path).unlink(missing_ok=True)
            except Exception:
                pass


async def rescore_miner_response(
    challenge_record: ChallengeRecord,
    challenge: Any,
    payload: Any,
    frame_store: Any,
    pseudo_gt_annotations: list,
    manifest: Manifest,
) -> float:
    settings = get_settings()
    miner_predictions = challenge_record.miner_predictions

    if not miner_predictions and challenge_record.responses_key:
        public_url = settings.R2_BUCKET_PUBLIC_URL
        miner_predictions = await fetch_miner_predictions(
            challenge_record.responses_key, public_url
        )

    if not miner_predictions:
        logger.warning("No miner predictions found for challenge %s", challenge_record.challenge_id)
        return 0.0

    logger.info("Rescoring with %d predicted frames", len(miner_predictions.get("frames", [])))

    miner_run = SVRunOutput(
        success=True,
        latency_ms=0.0,
        predictions=miner_predictions,
        error=None,
    )

    try:
        evaluation = post_vlm_ranking(
            payload=payload,
            miner_run=miner_run,
            challenge=challenge,
            pseudo_gt_annotations=pseudo_gt_annotations,
            frame_store=frame_store,
            manifest=manifest,
            element_id=challenge_record.element_id,
        )
        return evaluation.score
    except Exception as e:
        logger.error("Failed to score miner response: %s", e)
        return 0.0


async def run_spotcheck(
    challenge_record: ChallengeRecord,
    threshold: float | None = None,
    manifest: Manifest | None = None,
) -> SpotcheckResult:
    settings = get_settings()
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    logger.info(
        "Running spotcheck for challenge=%s miner=%s central_score=%.4f",
        challenge_record.challenge_id,
        challenge_record.miner_hotkey[:8] if challenge_record.miner_hotkey else "?",
        challenge_record.central_score,
    )

    if manifest is None:
        manifest = await load_manifest_for_spotcheck(challenge_record.block)
        if manifest is None:
            logger.error("Could not load manifest for spotcheck")
            return SpotcheckResult(
                challenge_id=challenge_record.challenge_id,
                element_id=challenge_record.element_id,
                miner_hotkey=challenge_record.miner_hotkey,
                central_score=challenge_record.central_score,
                audit_score=0.0,
                match_percentage=0.0,
                passed=False,
                details={"error": "manifest_load_failed"},
            )

    gt_result = await regenerate_ground_truth_sam3(
        challenge_record=challenge_record,
        manifest=manifest,
        required_n_frames=settings.SCOREVISION_VLM_SELECT_N_FRAMES,
    )

    if gt_result is None:
        logger.warning("Could not regenerate ground truth for spotcheck")
        return SpotcheckResult(
            challenge_id=challenge_record.challenge_id,
            element_id=challenge_record.element_id,
            miner_hotkey=challenge_record.miner_hotkey,
            central_score=challenge_record.central_score,
            audit_score=0.0,
            match_percentage=0.0,
            passed=False,
            details={"error": "ground_truth_generation_failed"},
        )

    challenge, payload, frame_store, pseudo_gt_annotations = gt_result

    audit_score = await rescore_miner_response(
        challenge_record=challenge_record,
        challenge=challenge,
        payload=payload,
        frame_store=frame_store,
        pseudo_gt_annotations=pseudo_gt_annotations,
        manifest=manifest,
    )

    match_pct = calculate_match_percentage(challenge_record.central_score, audit_score)
    passed = match_pct >= threshold

    if not passed:
        logger.warning(
            "SPOTCHECK FAILED: challenge=%s miner=%s central=%.4f audit=%.4f match=%.2f%%",
            challenge_record.challenge_id,
            challenge_record.miner_hotkey[:8] if challenge_record.miner_hotkey else "?",
            challenge_record.central_score,
            audit_score,
            match_pct * 100,
        )
    else:
        logger.info(
            "Spotcheck PASSED: challenge=%s miner=%s central=%.4f audit=%.4f match=%.2f%%",
            challenge_record.challenge_id,
            challenge_record.miner_hotkey[:8] if challenge_record.miner_hotkey else "?",
            challenge_record.central_score,
            audit_score,
            match_pct * 100,
        )

    try:
        frame_store.unlink()
    except Exception:
        pass

    return SpotcheckResult(
        challenge_id=challenge_record.challenge_id,
        element_id=challenge_record.element_id,
        miner_hotkey=challenge_record.miner_hotkey,
        central_score=challenge_record.central_score,
        audit_score=audit_score,
        match_percentage=match_pct,
        passed=passed,
        details={"n_pseudo_gt_frames": len(pseudo_gt_annotations)},
    )


async def spotcheck_loop(
    min_interval_seconds: int | None = None,
    max_interval_seconds: int | None = None,
    tail_blocks: int = 28800,
    threshold: float | None = None,
    element_id: str | None = None,
) -> None:
    settings = get_settings()

    if min_interval_seconds is None:
        min_interval_seconds = settings.AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if max_interval_seconds is None:
        max_interval_seconds = settings.AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    logger.info(
        "Starting spotcheck loop (interval=%d-%d seconds, threshold=%.0f%%, element=%s)",
        min_interval_seconds,
        max_interval_seconds,
        threshold * 100,
        element_id or "any",
    )

    manifest = await load_manifest_for_spotcheck()
    first_run = True

    while True:
        try:
            if first_run:
                first_run = False
                logger.info("[SpotcheckLoop] Running immediate first spotcheck")
            else:
                delay = calculate_next_spotcheck_delay(min_interval_seconds, max_interval_seconds)
                logger.info("[SpotcheckLoop] Next spotcheck in %.0f seconds", delay)
                await asyncio.sleep(delay)

            challenge_record = await fetch_random_challenge_record(tail_blocks, element_id=element_id)
            if challenge_record is None:
                logger.warning("[SpotcheckLoop] No challenge found for spotcheck")
                continue

            result = await run_spotcheck(challenge_record, threshold=threshold, manifest=manifest)
            logger.info(
                "[SpotcheckLoop] Spotcheck complete: passed=%s match=%.2f%% central=%.4f audit=%.4f",
                result.passed,
                result.match_percentage * 100,
                result.central_score,
                result.audit_score,
            )

        except asyncio.CancelledError:
            logger.info("[SpotcheckLoop] Cancelled, shutting down")
            break
        except Exception as e:
            logger.exception("[SpotcheckLoop] Error: %s", e)
            await asyncio.sleep(60)


def load_challenge_record_from_mock_dir(mock_data_dir: Path) -> ChallengeRecord | None:
    eval_files = list(mock_data_dir.glob("*_evaluation.json"))
    if not eval_files:
        logger.error("No *_evaluation.json found in %s", mock_data_dir)
        return None

    eval_file = eval_files[0]
    logger.info("Loading mock evaluation from: %s", eval_file)

    try:
        import json
        with open(eval_file) as f:
            eval_data = json.load(f)

        payload = eval_data.get("payload") or {}
        miner_info = payload.get("miner") or {}
        meta = payload.get("meta") or {}

        challenge_id = str(payload.get("task_id") or meta.get("task_id") or "mock")
        element_id = str(payload.get("element_id") or "")
        window_id = str(payload.get("window_id") or "")
        miner_hotkey = miner_info.get("hotkey") or ""
        central_score = float((payload.get("evaluation") or {}).get("score", 0.0))
        responses_key = (payload.get("run") or {}).get("responses_key")
        video_url = meta.get("video_url")
        scored_frame_numbers = payload.get("scored_frame_numbers")

        miner_predictions = None
        if responses_key:
            resp_file = mock_data_dir / Path(responses_key).name
            if resp_file.exists():
                logger.info("Loading mock responses from: %s", resp_file)
                with open(resp_file) as f:
                    resp_data = json.load(f)
                miner_predictions = resp_data.get("predictions")
                if not video_url:
                    video_url = resp_data.get("video_url")

        record = ChallengeRecord(
            challenge_id=challenge_id,
            element_id=element_id,
            window_id=window_id,
            block=0,
            miner_hotkey=miner_hotkey,
            central_score=central_score,
            payload=payload,
            miner_predictions=miner_predictions,
            video_url=video_url,
            responses_key=responses_key,
            scored_frame_numbers=scored_frame_numbers,
        )

        logger.info(
            "Loaded mock challenge: id=%s element=%s central_score=%.4f scored_frames=%s",
            record.challenge_id,
            record.element_id,
            record.central_score,
            record.scored_frame_numbers,
        )
        return record

    except Exception as e:
        logger.error("Failed to load mock data: %s", e)
        return None


async def run_single_spotcheck(
    tail_blocks: int = 28800,
    element_id: str | None = None,
    threshold: float | None = None,
    mock_data_dir: Path | None = None,
) -> SpotcheckResult | None:
    settings = get_settings()
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    if mock_data_dir is not None:
        logger.info("Using mock data from: %s", mock_data_dir)
        challenge_record = load_challenge_record_from_mock_dir(mock_data_dir)
    else:
        logger.info("Starting single spotcheck (tail=%d, element=%s)", tail_blocks, element_id or "any")
        challenge_record = await fetch_random_challenge_record(tail_blocks, element_id=element_id)

    if challenge_record is None:
        logger.error("No challenge found - check R2_BUCKET_PUBLIC_URL or mock data dir")
        return None

    logger.info("Found challenge: %s", challenge_record.challenge_id)

    manifest = await load_manifest_for_spotcheck(challenge_record.block)
    if manifest is None:
        logger.error("Could not load manifest")
        return None

    logger.info("Running spotcheck scoring...")
    result = await run_spotcheck(challenge_record, threshold=threshold, manifest=manifest)

    logger.info(
        "Spotcheck result: passed=%s match=%.2f%% central=%.4f audit=%.4f",
        result.passed,
        result.match_percentage * 100,
        result.central_score,
        result.audit_score,
    )

    return result
