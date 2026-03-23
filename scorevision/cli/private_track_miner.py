import logging
import os
from json import dumps
from pathlib import Path
import click
from scorevision.cli import console
from scorevision.cli.errors import ConfigError, DockerBuildError, DockerPushError, DockerRunError
from scorevision.utils.docker_helpers import DockerImage, build_image, get_image_digest, login_ghcr, push_image, run_container
from scorevision.utils.manifest import get_current_manifest, load_manifest_from_public_index
from scorevision.utils.settings import get_settings

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


def push_miner_image(image: DockerImage) -> str:
    username, token = get_ghcr_credentials()

    if not login_ghcr(username, token):
        raise DockerPushError("GHCR login failed")

    console.info(f"Pushing {image.full_name}")
    if not push_image(image):
        raise DockerPushError("Docker push failed")

    digest = get_image_digest(image)
    if digest:
        console.info(f"Image digest: {digest}")
    else:
        console.warn("Could not retrieve image digest — commit will omit it")
    console.success("Push complete\n")
    return digest


async def _resolve_private_element_id_from_manifest(
    element_id: str | None,
    *,
    skip_commit: bool,
) -> str | None:
    if element_id:
        return element_id
    if skip_commit:
        return None

    settings = get_settings()
    manifest = None

    if getattr(settings, "URL_MANIFEST", None):
        cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
        try:
            manifest = await load_manifest_from_public_index(
                settings.URL_MANIFEST,
                cache_dir=cache_dir,
            )
        except Exception as e:
            click.echo(f"Warning: unable to load manifest from URL_MANIFEST: {e}")

    if manifest is None:
        try:
            manifest = get_current_manifest()
        except Exception as e:
            raise click.ClickException(
                "Unable to load manifest. Configure URL_MANIFEST or SCOREVISION_MANIFEST_PATH/SV_MANIFEST_PATH."
            ) from e

    private_element_ids: list[str] = []
    for element in manifest.elements:
        eid = str(getattr(element, "id", "")).strip()
        if not eid:
            continue
        track = str(getattr(element, "track", "") or "").strip()
        if track == "private":
            private_element_ids.append(eid)

    private_element_ids = list(dict.fromkeys(private_element_ids))
    if not private_element_ids:
        raise click.ClickException("No private track element IDs found in the current manifest.")

    click.echo("Available private element IDs from manifest:")
    for idx, eid in enumerate(private_element_ids, start=1):
        click.echo(f"  {idx}. {eid}")

    choice = click.prompt(
        "Select private element ID (number)",
        type=click.IntRange(1, len(private_element_ids)),
    )
    return private_element_ids[choice - 1]


async def commit_on_chain(image: DockerImage, element_id: str, image_digest: str = "") -> None:
    from bittensor import wallet, async_subtensor

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
        "image_digest": image_digest,
        "element_id": str(element_id),
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
    settings = get_settings()
    port = int(os.environ.get("MINER_PORT", "8000"))
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    netuid = os.environ.get("NETUID") or str(settings.SCOREVISION_NETUID)

    # Map scorevision env names to the names expected by fiber.
    coldkey = os.environ.get("COLDKEY") or settings.BITTENSOR_WALLET_COLD
    hotkey = os.environ.get("HOTKEY") or settings.BITTENSOR_WALLET_HOT
    subtensor_network = os.environ.get("SUBTENSOR_NETWORK")
    subtensor_address = os.environ.get("SUBTENSOR_ADDRESS")
    endpoint = os.environ.get("BITTENSOR_SUBTENSOR_ENDPOINT") or settings.BITTENSOR_SUBTENSOR_ENDPOINT
    if not subtensor_network and not subtensor_address and endpoint:
        if endpoint in {"test", "finney", "local"}:
            subtensor_network = endpoint
        else:
            subtensor_address = endpoint

    env_vars = {
        "NETUID": netuid,
        "COLDKEY": coldkey,
        "HOTKEY": hotkey,
        "WALLET_NAME": coldkey,
        "HOTKEY_NAME": hotkey,
        "SUBTENSOR_NETWORK": subtensor_network,
        "SUBTENSOR_ADDRESS": subtensor_address,
    }
    env_vars = {k: v for k, v in env_vars.items() if v}
    volumes = [f"{settings.BITTENSOR_WALLET_PATH}:/root/.bittensor/wallets:ro"]

    console.info(f"Starting container on port {port}")
    container_id, error = run_container(
        image,
        port,
        detach=True,
        env_file=env_file,
        env_vars=env_vars,
        volumes=volumes,
    )

    if error:
        raise DockerRunError(f"Container failed to start:\n{error}")

    console.success(f"Miner running: {container_id[:12]}\n")


async def deploy_miner(
    tag: str,
    no_push: bool,
    no_commit: bool,
    no_start: bool,
    element_id: str | None,
) -> None:
    try:
        username, repo_name = get_miner_config()
        image = DockerImage(repository=f"{GHCR_REGISTRY}/{username}/{repo_name}", tag=tag)
        selected_element_id = await _resolve_private_element_id_from_manifest(
            element_id,
            skip_commit=no_commit,
        )

        build_miner_image(image)

        if not no_push:
            digest = push_miner_image(image)

            console.warn(
                "Remember to share your package with Score via GHCR package settings.\n"
                "See MINER.md for instructions.\n"
            )

            if not no_commit:
                if not selected_element_id:
                    raise click.ClickException("A private element_id is required for on-chain commit.")
                await commit_on_chain(image, selected_element_id, image_digest=digest)
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
