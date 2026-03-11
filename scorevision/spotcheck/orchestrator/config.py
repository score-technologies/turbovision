import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SpotcheckConfig:
    namespace: str
    runner_image: str
    dockerhub_secret: str
    runner_image_secret: str
    match_threshold: str
    miner_timeout_s: str
    job_active_deadline_s: int
    job_ttl_after_finished_s: int
    spotcheck_api_url: str
    gt_api_url: str
    blacklist_api_url: str
    netuid: int
    subtensor_network: str
    allowed_video_domains: str
    max_image_size_gb: float
    miner_ready_timeout_s: int
    max_spotcheck_retries: int
    auth_token: str
    test_mode: bool


def load_config() -> SpotcheckConfig:
    return SpotcheckConfig(
        namespace=os.environ.get("SPOTCHECK_NAMESPACE", "spotcheck"),
        runner_image=os.environ["RUNNER_IMAGE"],
        dockerhub_secret=os.environ.get("DOCKERHUB_SECRET", "dockerhub-creds"),
        runner_image_secret=os.environ.get("RUNNER_IMAGE_SECRET", "manakoai-registry"),
        match_threshold=os.environ.get("MATCH_THRESHOLD", "0.98"),
        miner_timeout_s=os.environ.get("MINER_TIMEOUT_S", "120"),
        job_active_deadline_s=int(os.environ.get("JOB_ACTIVE_DEADLINE_S", "600")),
        job_ttl_after_finished_s=int(os.environ.get("JOB_TTL_AFTER_FINISHED_S", "300")),
        spotcheck_api_url=os.environ["SPOTCHECK_API_URL"],
        gt_api_url=os.environ["GT_API_URL"],
        blacklist_api_url=os.environ["BLACKLIST_API_URL"],
        netuid=int(os.environ.get("NETUID", "44")),
        subtensor_network=os.environ.get("SUBTENSOR_NETWORK", "finney"),
        allowed_video_domains=os.environ.get("ALLOWED_VIDEO_DOMAINS", "scoredata.me"),
        max_image_size_gb=float(os.environ.get("MAX_IMAGE_SIZE_GB", "30")),
        miner_ready_timeout_s=int(os.environ.get("MINER_READY_TIMEOUT_S", "600")),
        max_spotcheck_retries=int(os.environ.get("MAX_SPOTCHECK_RETRIES", "1")),
        auth_token=os.environ["SPOTCHECK_AUTH_TOKEN"],
        test_mode=os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes"),
    )
