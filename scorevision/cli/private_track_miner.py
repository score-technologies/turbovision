import os
from pathlib import Path
from scorevision.cli import console
from scorevision.cli.errors import ConfigError, DockerBuildError, DockerPushError, DockerRunError
from scorevision.utils.docker_helpers import DockerImage, build_image, login_dockerhub, push_image, run_container


def get_miner_config() -> tuple[str, str]:
    username = os.environ.get("DOCKERHUB_USERNAME")
    repo_name = os.environ.get("DOCKERHUB_REPO", "pt-solution")

    if not username:
        raise ConfigError("DOCKERHUB_USERNAME required - see MINER.md")

    return username, repo_name


def get_dockerhub_credentials() -> tuple[str, str]:
    username = os.environ.get("DOCKERHUB_USERNAME")
    token = os.environ.get("DOCKERHUB_TOKEN")

    if not username or not token:
        raise ConfigError("DOCKERHUB_USERNAME and DOCKERHUB_TOKEN required")

    return username, token


def build_miner_image(image: DockerImage) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "scorevision/miner/private_track/Dockerfile"

    console.info(f"Building {image.full_name}")
    if not build_image(str(dockerfile), str(repo_root), image):
        raise DockerBuildError("Docker build failed")
    console.success("Build complete\n")


def setup_dockerhub_repo(username: str, token: str, repo_name: str) -> None:
    from scorevision.utils.dockerhub_helpers import create_private_repo, get_auth_token

    console.info(f"Ensuring repo exists: {username}/{repo_name}")
    auth_token = get_auth_token(username, token)
    if not auth_token:
        raise DockerPushError("DockerHub authentication failed")

    if not create_private_repo(auth_token, username, repo_name):
        raise DockerPushError(f"Failed to create/verify repo: {username}/{repo_name}")
    console.success("Repo ready\n")


def share_with_score(username: str, token: str, repo_name: str) -> None:
    from scorevision.utils.dockerhub_helpers import add_collaborator, get_auth_token, SCORE_DOCKERHUB_USER

    console.info(f"Adding {SCORE_DOCKERHUB_USER} as collaborator")
    auth_token = get_auth_token(username, token)
    if not auth_token:
        raise DockerPushError("DockerHub authentication failed")

    if not add_collaborator(auth_token, username, repo_name, SCORE_DOCKERHUB_USER, "read"):
        raise DockerPushError(f"Failed to add {SCORE_DOCKERHUB_USER} as collaborator")
    console.success(f"{SCORE_DOCKERHUB_USER} can now pull your images\n")


def push_miner_image(image: DockerImage) -> None:
    username, token = get_dockerhub_credentials()

    if not login_dockerhub(username, token):
        raise DockerPushError("DockerHub login failed")

    console.info(f"Pushing {image.full_name}")
    if not push_image(image):
        raise DockerPushError("Docker push failed")
    console.success("Push complete\n")


async def commit_on_chain(image: DockerImage) -> None:
    console.info("Committing on-chain")
    # TODO: Implement private track on-chain commit (register image_repo + image_tag)
    console.warn("Private track on-chain commit not yet implemented\n")


def start_miner_container(image: DockerImage) -> None:
    port = int(os.environ.get("MINER_PORT", "8000"))
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    console.info(f"Starting container on port {port}")
    container_id, error = run_container(image, port, detach=True, env_file=env_file)

    if error:
        raise DockerRunError(f"Container failed to start:\n{error}")

    console.success(f"Miner running: {container_id[:12]}\n")


async def deploy_miner(tag: str, no_push: bool, no_share: bool, no_commit: bool, no_start: bool) -> None:
    try:
        username, repo_name = get_miner_config()
        image = DockerImage(repository=f"{username}/{repo_name}", tag=tag)

        build_miner_image(image)

        if not no_push:
            dockerhub_username, dockerhub_token = get_dockerhub_credentials()

            setup_dockerhub_repo(dockerhub_username, dockerhub_token, repo_name)
            push_miner_image(image)

            if not no_share:
                share_with_score(dockerhub_username, dockerhub_token, repo_name)
            else:
                console.warn("Skipping Score collaborator share\n")

            if not no_commit:
                await commit_on_chain(image)
            else:
                console.warn("Skipping on-chain commit\n")

        if not no_start:
            start_miner_container(image)

        console.done()

    except ConfigError as e:
        console.error(str(e))
        raise SystemExit(1)
    except (DockerBuildError, DockerPushError, DockerRunError) as e:
        console.error(str(e))
        raise SystemExit(1)
