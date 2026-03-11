import json
import logging
import time
from kubernetes import client as k8s
from kubernetes.client.exceptions import ApiException

logger = logging.getLogger(__name__)

MAX_TRANSIENT_RETRIES = 5
TRANSIENT_BACKOFF_S = 5
MINER_READY_POLL_S = 5


def create_miner_job(batch_api: k8s.BatchV1Api, job: k8s.V1Job) -> str:
    batch_api.create_namespaced_job(job.metadata.namespace, job)
    logger.info("Created miner Job: %s", job.metadata.name)
    return job.metadata.name


def create_miner_service(core_api: k8s.CoreV1Api, service: k8s.V1Service) -> str:
    core_api.create_namespaced_service(service.metadata.namespace, service)
    logger.info("Created miner Service: %s", service.metadata.name)
    return service.metadata.name


def wait_for_miner_ready(
    core_api: k8s.CoreV1Api,
    spotcheck_id: str,
    namespace: str,
    timeout_s: int = 600,
) -> bool:
    deadline = time.monotonic() + timeout_s
    last_phase = None
    selector = f"spotcheck-id={spotcheck_id},role=miner"
    while time.monotonic() < deadline:
        try:
            phase = _get_miner_pod_phase(core_api, selector, namespace)
            if phase and phase != last_phase:
                elapsed = int(timeout_s - (deadline - time.monotonic()))
                logger.info("Miner pod phase: %s (%ds elapsed)", phase, elapsed)
                last_phase = phase

            pods = core_api.list_namespaced_pod(namespace, label_selector=selector)
            for pod in pods.items:
                if pod.status.phase == "Running":
                    statuses = pod.status.container_statuses or []
                    if statuses and statuses[0].ready:
                        logger.info("Miner pod %s is ready", pod.metadata.name)
                        return True
        except ApiException as exc:
            if exc.status >= 500:
                logger.warning("Transient API error checking miner readiness: %s", exc.reason)
            else:
                raise
        time.sleep(MINER_READY_POLL_S)
    logger.error("Miner not ready after %ds (last phase: %s)", timeout_s, last_phase)
    return False


def _get_miner_pod_phase(
    core_api: k8s.CoreV1Api, selector: str, namespace: str,
) -> str | None:
    try:
        pods = core_api.list_namespaced_pod(namespace, label_selector=selector)
        for pod in pods.items:
            statuses = pod.status.container_statuses or []
            if statuses and statuses[0].state:
                if statuses[0].state.waiting:
                    return statuses[0].state.waiting.reason or "Waiting"
                if statuses[0].state.running:
                    return "Running"
                if statuses[0].state.terminated:
                    return f"Terminated ({statuses[0].state.terminated.reason})"
            return pod.status.phase
    except Exception:
        pass
    return None


def cleanup_miner(
    batch_api: k8s.BatchV1Api,
    core_api: k8s.CoreV1Api,
    miner_job_name: str,
    service_name: str,
    namespace: str,
) -> None:
    try:
        batch_api.delete_namespaced_job(
            miner_job_name, namespace, propagation_policy="Background",
        )
        logger.info("Deleted miner Job: %s", miner_job_name)
    except Exception as exc:
        logger.warning("Failed to delete miner Job %s: %s", miner_job_name, exc)

    try:
        core_api.delete_namespaced_service(service_name, namespace)
        logger.info("Deleted miner Service: %s", service_name)
    except Exception as exc:
        logger.warning("Failed to delete miner Service %s: %s", service_name, exc)


def submit_job(batch_api: k8s.BatchV1Api, job: k8s.V1Job) -> str:
    batch_api.create_namespaced_job(job.metadata.namespace, job)
    logger.info("Created Job: %s", job.metadata.name)
    return job.metadata.name


def wait_for_completion(
    batch_api: k8s.BatchV1Api,
    job_name: str,
    namespace: str,
    timeout_s: int = 900,
) -> str:
    deadline = time.monotonic() + timeout_s
    transient_failures = 0
    while time.monotonic() < deadline:
        try:
            status = batch_api.read_namespaced_job(job_name, namespace).status
            transient_failures = 0
        except ApiException as exc:
            if exc.status >= 500 and transient_failures < MAX_TRANSIENT_RETRIES:
                transient_failures += 1
                logger.warning(
                    "Transient API error (%s), retry %d/%d",
                    exc.reason, transient_failures, MAX_TRANSIENT_RETRIES,
                )
                time.sleep(TRANSIENT_BACKOFF_S)
                continue
            raise
        if status.succeeded and status.succeeded > 0:
            return "Complete"
        if status.failed and status.failed > 0:
            return "Failed"
        time.sleep(10)
    return "Timeout"


def read_checker_result(
    core_api: k8s.CoreV1Api,
    spotcheck_id: str,
    namespace: str,
) -> dict | None:
    logs = _read_checker_logs(core_api, spotcheck_id, namespace)
    return _parse_last_json_line(logs)


def cleanup_job(batch_api: k8s.BatchV1Api, job_name: str, namespace: str) -> None:
    try:
        batch_api.delete_namespaced_job(
            job_name, namespace, propagation_policy="Background",
        )
    except Exception:
        pass


def gpu_node_exists(core_api: k8s.CoreV1Api) -> bool:
    try:
        nodes = core_api.list_node(label_selector="spotcheck-role=gpu-worker")
        return len(nodes.items) > 0
    except Exception:
        return False


def delete_gpu_nodes(core_api: k8s.CoreV1Api) -> None:
    nodes = core_api.list_node(label_selector="spotcheck-role=gpu-worker")
    for node in nodes.items:
        logger.info("Deleting GPU node: %s", node.metadata.name)
        try:
            core_api.delete_node(node.metadata.name)
        except Exception as e:
            logger.warning("Failed to delete node %s: %s", node.metadata.name, e)


def _read_checker_logs(
    core_api: k8s.CoreV1Api, spotcheck_id: str, namespace: str,
) -> str | None:
    selector = f"spotcheck-id={spotcheck_id},role=checker"
    pods = core_api.list_namespaced_pod(namespace, label_selector=selector)
    for pod in pods.items:
        try:
            return core_api.read_namespaced_pod_log(
                pod.metadata.name, namespace, container="checker",
            )
        except Exception as e:
            logger.warning("Failed to read logs from %s: %s", pod.metadata.name, e)
    return None


def _parse_last_json_line(logs: str | None) -> dict | None:
    if not logs:
        return None
    for line in reversed(logs.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return None
