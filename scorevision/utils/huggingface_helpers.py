from logging import getLogger
from pathlib import Path
from asyncio import Semaphore, gather, to_thread

from huggingface_hub import HfApi

from scorevision.utils.settings import get_settings


logger = getLogger(__name__)


def get_huggingface_repo_name() -> str:
    settings = get_settings()
    nickname = settings.BITTENSOR_WALLET_HOT
    return f"{settings.HUGGINGFACE_USERNAME}/ScoreVision-{nickname}"


def verify_huggingface_repo_name_exists(hf_api: HfApi) -> None:
    name = get_huggingface_repo_name()
    try:
        info = hf_api.repo_info(repo_id=name, repo_type="model")
    except Exception as e:
        raise ValueError(
            f"{e}.\n\nIf this is your first time, specify the path to the model to upload"
        )


def verify_huggingface_repo_revision_exists(revision: str, hf_api: HfApi) -> None:
    name = get_huggingface_repo_name()
    info = hf_api.repo_info(repo_id=name, repo_type="model", revision=revision)
    logger.info(f"Repo Info:{info}")
    revision_ = getattr(info, "sha", None) or getattr(info, "oid", None)
    if revision != revision_:
        raise ValueError(
            f"HF revision not accessible (gated/missing?): {revision_} != {revision}"
        )


def get_paths_in_directory(path_dir: Path) -> list[Path]:
    def is_hidden(path: Path) -> bool:
        return any(
            part.startswith(".") and part not in (".", "..") for part in path.parts
        )

    def is_lock(path: Path) -> bool:
        return path.name.startswith(".") or path.name.endswith(".lock")

    paths = []
    for path in path_dir.rglob("*"):
        if not path.is_file():
            continue
        if is_hidden(path=path):
            continue
        if is_lock(path=path):
            continue
        paths.append(path)
    logger.info(f"{len(paths)} files found")
    return paths


async def upload_file_to_huggingface_repo(
    name: str, path_file: Path, path_dir: Path, semaphore: Semaphore, hf_api: HfApi
) -> None:
    async with semaphore:
        await to_thread(
            lambda: hf_api.upload_file(
                path_or_fileobj=str(path_file),
                path_in_repo=str(path_file.relative_to(path_dir)),
                repo_id=name,
                repo_type="model",
                commit_message="scorevision: push artifact",
            )
        )


async def upload_directory_to_huggingface_repo(path_dir: Path, hf_api: HfApi) -> None:
    logger.info(f"Uploading {path_dir}")
    settings = get_settings()
    semaphore = Semaphore(settings.HUGGINGFACE_CONCURRENCY)
    repo_name = get_huggingface_repo_name()
    paths = get_paths_in_directory(path_dir=path_dir)
    await gather(
        *(
            upload_file_to_huggingface_repo(
                name=repo_name,
                path_file=path,
                path_dir=path_dir,
                semaphore=semaphore,
                hf_api=hf_api,
            )
            for path in paths
        )
    )


async def create_or_update_huggingface_repo(model_path: Path, hf_api: HfApi) -> None:
    name = get_huggingface_repo_name()
    hf_api.create_repo(repo_id=name, repo_type="model", private=True, exist_ok=True)

    try:
        hf_api.update_repo_visibility(repo_id=name, private=True)
    except Exception as e:
        logger.error(f"Error making hf repo private: {e}")

    await upload_directory_to_huggingface_repo(path_dir=model_path, hf_api=hf_api)


async def get_huggingface_repo_revision(hf_api: HfApi) -> str:
    name = get_huggingface_repo_name()
    info = hf_api.repo_info(repo_id=name, repo_type="model")
    revision = getattr(info, "sha", getattr(info, "oid", "")) or ""
    logger.info(f"Detected revision: {revision}")
    return revision


async def create_update_or_verify_huggingface_repo(
    model_path: Path | None, hf_revision: str | None
) -> str:
    """
    if model_path is provided, the huggingface repo will be created or updated (if it already exists)
    if hf_revision is provided, the huggingface repo revision with be verified but not updated
    if model_path and hf_revision are both not provided,
        if a repo exists for the user, the latest revision will be used
        otherwise: an error will be thrown asking the user to specify a path to a model for upload
    """
    settings = get_settings()
    if (
        not settings.HUGGINGFACE_USERNAME
        and not settings.HUGGINGFACE_API_KEY.get_secret_value()
    ):
        raise ValueError("HUGGINGFACE_USERNAME/HUGGINGFACE_API_KEY required")
    hf_api = HfApi(token=settings.HUGGINGFACE_API_KEY.get_secret_value())

    if model_path:
        logger.info(f"Creating/Updating repo")
        await create_or_update_huggingface_repo(model_path=model_path, hf_api=hf_api)
    else:
        verify_huggingface_repo_name_exists(hf_api=hf_api)
        logger.info(f"Using existing repo")

    if hf_revision:
        verify_huggingface_repo_revision_exists(revision=hf_revision, hf_api=hf_api)
        logger.info(f"Using provided revision: {hf_revision}")
    else:
        hf_revision = await get_huggingface_repo_revision(hf_api=hf_api)
        logger.info(f"Hf revision: {hf_revision}")

    try:
        hf_api.update_repo_settings(
            repo_id=get_huggingface_repo_name(), repo_type="model", private=False
        )
    except Exception as e:
        logger.error(f"Error making hf repo public: {e}")
        pass

    return hf_revision
