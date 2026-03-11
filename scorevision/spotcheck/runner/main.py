import asyncio
import json
import sys
from scorevision.spotcheck.runner.config import RunnerConfig, load_config
from scorevision.spotcheck.runner.miner_client import send_challenge, wait_for_ready
from scorevision.spotcheck.runner.url_validator import is_video_url_allowed
from scorevision.validator.central.private_track.scoring import score_predictions


def emit(result: dict) -> None:
    print(json.dumps(result), flush=True)


def scores_match(original: float, spotcheck: float, threshold: float) -> bool:
    tolerance = 1.0 - threshold
    return abs(original - spotcheck) <= tolerance


async def run(cfg: RunnerConfig) -> None:
    print(
        f"Spotcheck: challenge={cfg.challenge_id} "
        f"miner={cfg.miner_hotkey} original_score={cfg.original_score:.4f}",
        flush=True,
    )

    base = {
        "miner_hotkey": cfg.miner_hotkey,
        "challenge_id": cfg.challenge_id,
        "original_score": cfg.original_score,
    }

    if not is_video_url_allowed(cfg.video_url, cfg.allowed_video_domains):
        print(f"Rejected video URL (domain not allowed): {cfg.video_url}", flush=True)
        emit({**base, "status": "FAIL", "reason": "video_url_rejected"})
        sys.exit(1)

    print(f"Waiting for miner at {cfg.miner_url}...", flush=True)
    if not await wait_for_ready(cfg.miner_url):
        emit({**base, "status": "FAIL", "reason": "miner_not_ready"})
        sys.exit(1)

    print("Miner ready. Sending challenge...", flush=True)
    response, elapsed, timed_out = await send_challenge(
        cfg.challenge_id, cfg.video_url, cfg.miner_timeout_s, cfg.miner_url,
    )

    if timed_out:
        emit({**base, "status": "FAIL", "reason": "timeout", "elapsed_s": round(elapsed, 3)})
        sys.exit(1)

    if response is None:
        emit({**base, "status": "FAIL", "reason": "no_response"})
        sys.exit(1)

    spotcheck_score = score_predictions(response.predictions, cfg.ground_truth)

    passed = scores_match(cfg.original_score, spotcheck_score, cfg.match_threshold)

    emit({
        **base,
        "status": "PASS" if passed else "FAIL",
        "reason": "score_match" if passed else "score_mismatch",
        "spotcheck_score": round(spotcheck_score, 6),
        "score_diff": round(abs(cfg.original_score - spotcheck_score), 6),
        "elapsed_s": round(elapsed, 3),
        "prediction_count": len(response.predictions),
        "processing_time": round(response.processing_time, 3),
    })

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(run(load_config()))
