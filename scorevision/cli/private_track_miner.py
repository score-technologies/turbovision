import logging
import os
from json import dumps
from pathlib import Path
from scorevision.cli import console
from scorevision.cli.errors import ConfigError, DockerBuildError, DockerPushError, DockerRunError
from scorevision.utils.docker_helpers import DockerImage, build_image, login_ghcr, push_image, run_container

logger = logging.getLogger(__name__)

GHCR_REGISTRY = "ghcr.io"


def get_miner_config() -> tuple[str, str]:
    username = os.environ.get("GITHUB_USERNAME")
    repo_name = os.environ.get("GHCR_REPO", "pt-solution")

    if not username:
        raise ConfigError("GITHUB_USERNAME required - see MINER.md")

    return username, repo_name


def get_ghcr_credentials() -> tuple[str, str]:
    username = os.environ.get("GITHUB_USERNAME")
    token = os.environ.get("GITHUB_TOKEN")

    if not username or not token:
        raise ConfigError("GITHUB_USERNAME and GITHUB_TOKEN required")

    return username, token


def build_miner_image(image: DockerImage) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "scorevision/miner/private_track/Dockerfile"

    console.info(f"Building {image.full_name}")
    if not build_image(str(dockerfile), str(repo_root), image):
        raise DockerBuildError("Docker build failed")
    console.success("Build complete\n")


def push_miner_image(image: DockerImage) -> None:
    username, token = get_ghcr_credentials()

    if not login_ghcr(username, token):
        raise DockerPushError("GHCR login failed")

    console.info(f"Pushing {image.full_name}")
    if not push_image(image):
        raise DockerPushError("Docker push failed")
    console.success("Push complete\n")


async def commit_on_chain(image: DockerImage) -> None:
    from bittensor import wallet, async_subtensor
    from scorevision.utils.settings import get_settings

    console.info("Committing on-chain")
    settings = get_settings()
    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    payload = {
        "role": "miner",
        "track": "private",
        "image_repo": image.repository,
        "image_tag": image.tag,
        "hotkey": w.hotkey.ss58_address,
    }
    logger.info("Commit payload: %s", payload)
    try:
        sub = async_subtensor(settings.BITTENSOR_SUBTENSOR_ENDPOINT)
        await sub.initialize()
        await sub.set_reveal_commitment(
            wallet=w,
            netuid=settings.SCOREVISION_NETUID,
            data=dumps(payload),
            blocks_until_reveal=1,
        )
        console.success("On-chain commitment submitted\n")
    except Exception as e:
        logger.error("On-chain commit failed: %s: %s", type(e).__name__, e)
        console.warn(f"On-chain commit failed: {e}\n")


def start_miner_container(image: DockerImage) -> None:
    port = int(os.environ.get("MINER_PORT", "8000"))
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    console.info(f"Starting container on port {port}")
    container_id, error = run_container(image, port, detach=True, env_file=env_file)

    if error:
        raise DockerRunError(f"Container failed to start:\n{error}")

    console.success(f"Miner running: {container_id[:12]}\n")


async def deploy_miner(tag: str, no_push: bool, no_commit: bool, no_start: bool) -> None:
    try:
        username, repo_name = get_miner_config()
        image = DockerImage(repository=f"{GHCR_REGISTRY}/{username}/{repo_name}", tag=tag)

        build_miner_image(image)

        if not no_push:
            push_miner_image(image)

            console.warn(
                "Remember to share your package with Score via GHCR package settings.\n"
                "See MINER.md for instructions.\n"
            )

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
