import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CONTAINER_STARTUP_WAIT = 3


@dataclass
class DockerImage:
    repository: str
    tag: str

    @property
    def full_name(self) -> str:
        return f"{self.repository}:{self.tag}"


def build_image(dockerfile_path: str, context_path: str, image: DockerImage) -> bool:
    logger.info("Building Docker image: %s", image.full_name)
    result = subprocess.run(
        ["docker", "build", "-f", dockerfile_path, "-t", image.full_name, context_path],
    )
    if result.returncode != 0:
        logger.error("Docker build failed")
        return False
    return True


def push_image(image: DockerImage) -> bool:
    logger.info("Pushing Docker image: %s", image.full_name)
    result = subprocess.run(["docker", "push", image.full_name])
    if result.returncode != 0:
        logger.error("Docker push failed")
        return False
    return True


def pull_image(image: DockerImage, timeout: float = 300) -> bool:
    logger.info("Pulling Docker image: %s", image.full_name)
    result = subprocess.run(["docker", "pull", image.full_name], timeout=timeout)
    if result.returncode != 0:
        logger.error("Docker pull failed")
        return False
    return True


def is_container_running(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.decode().strip() == "true"


def get_container_logs(container_id: str, tail: int = 50) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_id],
        capture_output=True,
    )
    stderr = result.stderr.decode() if result.stderr else ""
    stdout = result.stdout.decode() if result.stdout else ""
    return stderr + stdout


def _remove_container(container_id: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


def run_container(
    image: DockerImage,
    port: int,
    detach: bool = True,
    env_file: Path | None = None,
    env_vars: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    logger.info("Running container: %s on port %d", image.full_name, port)
    cmd = ["docker", "run", "-p", f"{port}:{port}"]

    if env_file and env_file.exists():
        cmd.extend(["--env-file", str(env_file)])

    if env_vars:
        for key, value in env_vars.items():
            if value is not None:
                cmd.extend(["-e", f"{key}={value}"])

    if detach:
        cmd.append("-d")

    cmd.append(image.full_name)

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None, result.stderr.decode()

    container_id = result.stdout.decode().strip()

    if detach:
        time.sleep(_CONTAINER_STARTUP_WAIT)
        if not is_container_running(container_id):
            logs = get_container_logs(container_id)
            _remove_container(container_id)
            return None, f"Container exited immediately.\n\nLogs:\n{logs}"

    return container_id, None


def stop_container(container_id: str, remove: bool = True) -> bool:
    result = subprocess.run(
        ["docker", "stop", container_id],
        capture_output=True,
        timeout=30,
    )
    if result.returncode == 0 and remove:
        _remove_container(container_id)
    return result.returncode == 0


def login_dockerhub(username: str, token: str) -> bool:
    result = subprocess.run(
        ["docker", "login", "-u", username, "--password-stdin"],
        input=token.encode(),
        capture_output=True,
    )
    return result.returncode == 0
