import hashlib
import time
from kubernetes import client as k8s
from scorevision.spotcheck.orchestrator.config import SpotcheckConfig


def generate_spotcheck_id(hotkey: str, challenge_id: str) -> str:
    suffix = hashlib.sha256(
        f"{hotkey}-{challenge_id}-{time.time()}".encode()
    ).hexdigest()[:8]
    return suffix


def _build_image_reference(image_repo: str, image_tag: str, image_digest: str) -> str:
    if image_digest:
        return f"{image_repo}@{image_digest}"
    return f"{image_repo}:{image_tag}"


def build_miner_job(
    cfg: SpotcheckConfig,
    spotcheck_id: str,
    image_repo: str,
    image_tag: str,
    image_digest: str,
    miner_hotkey: str,
) -> k8s.V1Job:
    name = f"spotcheck-miner-{spotcheck_id}"
    image_ref = _build_image_reference(image_repo, image_tag, image_digest)
    labels = {
        "app": "spotcheck",
        "role": "miner",
        "spotcheck-id": spotcheck_id,
        "miner-hotkey": miner_hotkey[:16],
    }

    container = k8s.V1Container(
        name="miner",
        image=image_ref,
        ports=[k8s.V1ContainerPort(container_port=8000)],
        env=[
            k8s.V1EnvVar(name="VERIFY_ENABLED", value="false"),
            k8s.V1EnvVar(name="BLACKLIST_ENABLED", value="false"),
        ],
        resources=k8s.V1ResourceRequirements(
            requests={
                "nvidia.com/gpu": "1",
                "cpu": "4",
                "memory": "16Gi",
                "ephemeral-storage": "10Gi",
            },
            limits={
                "nvidia.com/gpu": "1",
                "cpu": "14",
                "memory": "180Gi",
                "ephemeral-storage": "50Gi",
            },
        ),
        readiness_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/docs", port=8000),
            initial_delay_seconds=10,
            period_seconds=5,
            failure_threshold=50,
        ),
        security_context=k8s.V1SecurityContext(
            run_as_non_root=False,
            allow_privilege_escalation=False,
            read_only_root_filesystem=False,
            capabilities=k8s.V1Capabilities(drop=["ALL"]),
        ),
    )

    pod_spec = k8s.V1PodSpec(
        containers=[container],
        restart_policy="OnFailure",
        automount_service_account_token=False,
        image_pull_secrets=[k8s.V1LocalObjectReference(name=cfg.dockerhub_secret)],
        tolerations=[
            k8s.V1Toleration(
                key="spotcheck-gpu", operator="Equal",
                value="true", effect="NoSchedule",
            ),
            k8s.V1Toleration(
                key="nvidia.com/gpu", operator="Exists",
                effect="NoSchedule",
            ),
        ],
        node_selector={"spotcheck-role": "gpu-worker"},
    )

    return k8s.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s.V1ObjectMeta(
            name=name,
            namespace=cfg.namespace,
            labels=labels,
        ),
        spec=k8s.V1JobSpec(
            backoff_limit=3,
            active_deadline_seconds=cfg.miner_ready_timeout_s,
            ttl_seconds_after_finished=cfg.job_ttl_after_finished_s,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels=labels),
                spec=pod_spec,
            ),
        ),
    )


def build_miner_service(
    cfg: SpotcheckConfig,
    spotcheck_id: str,
) -> k8s.V1Service:
    name = f"spotcheck-miner-{spotcheck_id}"
    return k8s.V1Service(
        api_version="v1",
        kind="Service",
        metadata=k8s.V1ObjectMeta(
            name=name,
            namespace=cfg.namespace,
            labels={
                "app": "spotcheck",
                "spotcheck-id": spotcheck_id,
            },
        ),
        spec=k8s.V1ServiceSpec(
            type="ClusterIP",
            selector={"spotcheck-id": spotcheck_id, "role": "miner"},
            ports=[k8s.V1ServicePort(port=8000, target_port=8000, protocol="TCP")],
        ),
    )


def build_checker_job(
    cfg: SpotcheckConfig,
    spotcheck_id: str,
    challenge_id: str,
    video_url: str,
    ground_truth_json: str,
    original_score: float,
    miner_hotkey: str,
) -> k8s.V1Job:
    job_name = f"spotcheck-checker-{spotcheck_id}"
    miner_service_url = f"http://spotcheck-miner-{spotcheck_id}:8000"
    labels = {
        "app": "spotcheck",
        "role": "checker",
        "spotcheck-id": spotcheck_id,
        "miner-hotkey": miner_hotkey[:16],
    }

    container = k8s.V1Container(
        name="checker",
        image=cfg.runner_image,
        env=[
            k8s.V1EnvVar(name="MINER_URL", value=miner_service_url),
            k8s.V1EnvVar(name="CHALLENGE_ID", value=challenge_id),
            k8s.V1EnvVar(name="VIDEO_URL", value=video_url),
            k8s.V1EnvVar(name="GROUND_TRUTH_JSON", value=ground_truth_json),
            k8s.V1EnvVar(name="ORIGINAL_SCORE", value=str(original_score)),
            k8s.V1EnvVar(name="MINER_HOTKEY", value=miner_hotkey),
            k8s.V1EnvVar(name="MATCH_THRESHOLD", value=cfg.match_threshold),
            k8s.V1EnvVar(name="MINER_TIMEOUT_S", value=cfg.miner_timeout_s),
            k8s.V1EnvVar(name="ALLOWED_VIDEO_DOMAINS", value=f"{cfg.allowed_video_domains},video-proxy"),
        ],
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"},
        ),
    )

    pod_spec = k8s.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        node_selector={"spotcheck-role": "system"},
        image_pull_secrets=[k8s.V1LocalObjectReference(name=cfg.runner_image_secret)],
        affinity=k8s.V1Affinity(
            pod_anti_affinity=k8s.V1PodAntiAffinity(
                required_during_scheduling_ignored_during_execution=[
                    k8s.V1PodAffinityTerm(
                        label_selector=k8s.V1LabelSelector(
                            match_labels={"role": "miner"},
                        ),
                        topology_key="kubernetes.io/hostname",
                    ),
                ],
            ),
        ),
    )

    return k8s.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s.V1ObjectMeta(
            name=job_name,
            namespace=cfg.namespace,
            labels=labels,
        ),
        spec=k8s.V1JobSpec(
            backoff_limit=0,
            active_deadline_seconds=cfg.job_active_deadline_s,
            ttl_seconds_after_finished=cfg.job_ttl_after_finished_s,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels=labels),
                spec=pod_spec,
            ),
        ),
    )
