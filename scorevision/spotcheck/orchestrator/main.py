import asyncio
import json
import logging
from urllib.parse import urlparse, urlunparse
from kubernetes import client as k8s, config as k8s_config
from scorevision.utils.docker_hub import check_image_accessible, exceeds_size_limit, fetch_image_size_gb
from scorevision.validator.central.private_track.scoring import PRIVATE_SCORING_VERSION
from scorevision.spotcheck.orchestrator.blacklist_client import report_failed_spotcheck
from scorevision.spotcheck.orchestrator.config import SpotcheckConfig, load_config
from scorevision.spotcheck.orchestrator.ground_truth_client import fetch_ground_truth
from scorevision.spotcheck.orchestrator.job_executor import (
    cleanup_job,
    cleanup_miner,
    create_miner_job,
    create_miner_service,
    delete_gpu_nodes,
    gpu_node_exists,
    read_checker_result,
    submit_job,
    wait_for_completion,
    wait_for_miner_ready,
)
from scorevision.spotcheck.orchestrator.job_factory import (
    build_checker_job,
    build_miner_job,
    build_miner_service,
    generate_spotcheck_id,
)
from scorevision.spotcheck.orchestrator.miner_registry import fetch_registered_miners
from scorevision.spotcheck.orchestrator.pending_spotcheck_client import fetch_pending_spotchecks, remove_completed_spotcheck

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("orchestrator")

INFRASTRUCTURE_REASONS = frozenset({
    "miner_not_ready",
    "spotcheck_job_timeout",
    "infrastructure_failure",
})


async def run(cfg: SpotcheckConfig) -> None:
    targets = await _resolve_targets(cfg)
    if not targets:
        logger.warning("No valid spotcheck targets found")
        return

    logger.info("Selected %d spotcheck targets", len(targets))

    batch_api, core_api = _connect_k8s()

    for i, target in enumerate(targets):
        logger.info(
            "[%d/%d] %s (image=%s:%s score=%.4f)",
            i + 1, len(targets),
            target["miner_hotkey"][:16],
            target["image_repo"], target["image_tag"],
            target["original_score"],
        )

        scoring_version = target.get("scoring_version", 0)
        if scoring_version != PRIVATE_SCORING_VERSION:
            logger.info(
                "Skipping %s — scoring version mismatch (result=%d, current=%d)",
                target["miner_hotkey"][:16], scoring_version, PRIVATE_SCORING_VERSION,
            )
            continue

        await _run_single_spotcheck(cfg, batch_api, core_api, target)
        delete_gpu_nodes(core_api)

    logger.info("Spotcheck batch complete")


async def _resolve_targets(cfg: SpotcheckConfig) -> list[dict]:
    pending = await fetch_pending_spotchecks(cfg.spotcheck_api_url, cfg.auth_token)
    if not pending:
        return []

    if cfg.test_mode:
        logger.info("Test mode — skipping miner registration check, using all %d pending targets", len(pending))
        targets = pending
    else:
        miners = await fetch_registered_miners(cfg.netuid, cfg.subtensor_network)
        alive_hotkeys = set(miners.keys())

        targets = []
        for target in pending:
            hotkey = target["miner_hotkey"]
            if hotkey not in alive_hotkeys:
                logger.info("Skipping %s — miner no longer registered", hotkey[:16])
                continue
            targets.append(target)

        logger.info("Pending: %d | Alive miners: %d | Eligible: %d", len(pending), len(alive_hotkeys), len(targets))

    if len(targets) > cfg.max_spotchecks_per_run:
        logger.info("Capping to %d oldest spotchecks (total eligible: %d)", cfg.max_spotchecks_per_run, len(targets))
        targets = targets[:cfg.max_spotchecks_per_run]

    return targets


async def _run_single_spotcheck(
    cfg: SpotcheckConfig,
    batch_api: k8s.BatchV1Api,
    core_api: k8s.CoreV1Api,
    target: dict,
) -> None:
    hotkey_short = target["miner_hotkey"][:16]
    challenge_id = target["challenge_id"]

    image_ref = target.get("image_digest") or target["image_tag"]

    accessible = await check_image_accessible(target["image_repo"], image_ref, ghcr_pat=cfg.ghcr_token)
    if not accessible:
        logger.warning("FAILED %s — image not accessible (miner has not shared GHCR package with Score)", hotkey_short)
        if not cfg.test_mode:
            await report_failed_spotcheck(cfg.blacklist_api_url, target["miner_hotkey"], "image_unavailable", cfg.auth_token)
            await remove_completed_spotcheck(cfg.spotcheck_api_url, challenge_id, cfg.auth_token)
        else:
            logger.info("Test mode — skipping blacklist report and removal for %s", hotkey_short)
        return

    image_size_gb = await fetch_image_size_gb(target["image_repo"], image_ref, ghcr_pat=cfg.ghcr_token)
    if image_size_gb is not None:
        logger.info("Image %s size: %.2f GB", image_ref, image_size_gb)
    if exceeds_size_limit(image_size_gb, cfg.max_image_size_gb):
        logger.warning("REJECTED %s — image %.2f GB exceeds limit %.0f GB", hotkey_short, image_size_gb, cfg.max_image_size_gb)
        if not cfg.test_mode:
            await report_failed_spotcheck(cfg.blacklist_api_url, target["miner_hotkey"], "image_too_large", cfg.auth_token)
            await remove_completed_spotcheck(cfg.spotcheck_api_url, challenge_id, cfg.auth_token)
        else:
            logger.info("Test mode — skipping blacklist report and removal for %s", hotkey_short)
        return

    ground_truth = await fetch_ground_truth(cfg.gt_api_url, target["challenge_id"], cfg.auth_token)
    if not ground_truth:
        logger.warning("Skipping %s — no ground truth", hotkey_short)
        return

    ground_truth_json = json.dumps(ground_truth, separators=(",", ":"))

    for attempt in range(1 + cfg.max_spotcheck_retries):
        if attempt > 0:
            logger.info("Retry %d/%d for %s", attempt, cfg.max_spotcheck_retries, hotkey_short)

        reason = await _attempt_spotcheck(
            cfg, batch_api, core_api, target, ground_truth_json,
        )

        if reason is None:
            logger.info("PASSED: %s", hotkey_short)
            if not cfg.test_mode:
                await remove_completed_spotcheck(cfg.spotcheck_api_url, challenge_id, cfg.auth_token)
            else:
                logger.info("Test mode — skipping removal for %s", hotkey_short)
            return

        is_infra_failure = reason in INFRASTRUCTURE_REASONS or _is_gpu_preemption(core_api, reason)

        if reason == "image_pull_failed":
            logger.warning("FAILED: %s — image no longer available", hotkey_short)
            if not cfg.test_mode:
                await report_failed_spotcheck(cfg.blacklist_api_url, target["miner_hotkey"], "image_unavailable", cfg.auth_token)
                await remove_completed_spotcheck(cfg.spotcheck_api_url, challenge_id, cfg.auth_token)
            else:
                logger.info("Test mode — skipping blacklist report and removal for %s", hotkey_short)
            return

        if is_infra_failure and attempt < cfg.max_spotcheck_retries:
            logger.warning("Infrastructure failure (%s), will retry %s", reason, hotkey_short)
            continue

        if is_infra_failure:
            logger.warning("Infrastructure failure (%s) after retries, skipping blacklist for %s", reason, hotkey_short)
        else:
            logger.warning("FAILED: %s — %s", hotkey_short, reason)
            if not cfg.test_mode:
                await report_failed_spotcheck(cfg.blacklist_api_url, target["miner_hotkey"], reason, cfg.auth_token)
            else:
                logger.info("Test mode — skipping blacklist report for %s", hotkey_short)

        if not cfg.test_mode:
            await remove_completed_spotcheck(cfg.spotcheck_api_url, challenge_id, cfg.auth_token)
        else:
            logger.info("Test mode — skipping removal for %s", hotkey_short)
        return


async def _attempt_spotcheck(
    cfg: SpotcheckConfig,
    batch_api: k8s.BatchV1Api,
    core_api: k8s.CoreV1Api,
    target: dict,
    ground_truth_json: str,
) -> str | None:
    spotcheck_id = generate_spotcheck_id(target["miner_hotkey"], target["challenge_id"])

    miner_video_url = _rewrite_video_url(target["video_url"])

    miner_job = build_miner_job(
        cfg,
        spotcheck_id=spotcheck_id,
        image_repo=target["image_repo"],
        image_tag=target["image_tag"],
        image_digest=target.get("image_digest", ""),
        miner_hotkey=target["miner_hotkey"],
    )
    service = build_miner_service(cfg, spotcheck_id)
    checker_job = build_checker_job(
        cfg,
        spotcheck_id=spotcheck_id,
        challenge_id=target["challenge_id"],
        video_url=miner_video_url,
        ground_truth_json=ground_truth_json,
        original_score=target["original_score"],
        miner_hotkey=target["miner_hotkey"],
    )

    miner_job_name = miner_job.metadata.name
    service_name = service.metadata.name
    checker_job_name = None

    try:
        create_miner_job(batch_api, miner_job)
        create_miner_service(core_api, service)

        if not wait_for_miner_ready(core_api, spotcheck_id, cfg.namespace, cfg.miner_ready_timeout_s):
            if _is_image_pull_failure(core_api, spotcheck_id, cfg.namespace):
                return "image_pull_failed"
            return "miner_not_ready"

        checker_job_name = submit_job(batch_api, checker_job)
        status = wait_for_completion(batch_api, checker_job_name, cfg.namespace)
        result = read_checker_result(core_api, spotcheck_id, cfg.namespace)

        if result:
            logger.info("Result: %s", json.dumps(result))

        if status == "Complete" and result and result.get("status") == "PASS":
            return None

        return _failure_reason(status, result)
    finally:
        if checker_job_name:
            cleanup_job(batch_api, checker_job_name, cfg.namespace)
        cleanup_miner(batch_api, core_api, miner_job_name, service_name, cfg.namespace)


def _is_image_pull_failure(
    core_api: k8s.CoreV1Api, spotcheck_id: str, namespace: str,
) -> bool:
    selector = f"spotcheck-id={spotcheck_id},role=miner"
    try:
        pods = core_api.list_namespaced_pod(namespace, label_selector=selector)
        for pod in pods.items:
            statuses = pod.status.container_statuses or []
            for status in statuses:
                if status.state and status.state.waiting:
                    reason = status.state.waiting.reason or ""
                    if reason in ("ErrImagePull", "ImagePullBackOff"):
                        return True
    except Exception:
        pass
    return False


def _is_gpu_preemption(core_api: k8s.CoreV1Api, reason: str) -> bool:
    if reason not in ("miner_not_ready", "spotcheck_failed", "spotcheck_job_timeout"):
        return False
    return not gpu_node_exists(core_api)


def _failure_reason(status: str, result: dict | None) -> str:
    if result and result.get("reason"):
        return f"spotcheck_{result['reason']}"
    if status == "Timeout":
        return "spotcheck_job_timeout"
    return "spotcheck_failed"


def _rewrite_video_url(original_url: str) -> str:
    parsed = urlparse(original_url)
    return urlunparse(parsed._replace(scheme="http", netloc="video-proxy"))


def _connect_k8s() -> tuple[k8s.BatchV1Api, k8s.CoreV1Api]:
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    return k8s.BatchV1Api(), k8s.CoreV1Api()


if __name__ == "__main__":
    asyncio.run(run(load_config()))
