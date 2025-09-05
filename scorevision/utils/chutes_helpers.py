from logging import getLogger
from json import loads, JSONDecodeError
from os import environ, chmod, stat
from stat import S_IEXEC
from asyncio import create_subprocess_exec, subprocess
from pathlib import Path
from contextlib import contextmanager
from random import Random

from jinja2 import Template
import petname

from scorevision.utils.settings import get_settings
from scorevision.utils.huggingface_helpers import get_huggingface_repo_name

from scorevision.utils.async_clients import get_async_client

logger = getLogger(__name__)


@contextmanager
def temporary_chutes_config_file(prefix: str, delete: bool = True):
    settings = get_settings()
    tmp_path = settings.PATH_CHUTE_TEMPLATES / f"{prefix}.py"
    try:
        with open(tmp_path, "w+") as f:
            yield f, tmp_path
    finally:
        if delete:
            tmp_path.unlink(missing_ok=True)


def generate_nickname(key: str) -> str:
    petname.random = Random(int(key, 16))
    return petname.Generate(words=2, separator="-")


def get_chute_name(hf_revision: str) -> str:
    settings = get_settings()
    nickname = generate_nickname(key=hf_revision)
    logger.info(f"Hf Revision ({hf_revision}) -> Nickname ({nickname})")
    return f"{settings.HUGGINGFACE_USERNAME.replace('/','-')}-{nickname}".lower()


def guess_chute_slug(hf_revision: str) -> str:
    settings = get_settings()
    chute_username = settings.CHUTES_USERNAME.replace("_", "-")
    chute_name = get_chute_name(hf_revision=hf_revision)
    return f"{chute_username}-{chute_name}"


def render_chute_template(
    revision: str,
) -> Template:

    settings = get_settings()
    hf_repo_name = get_huggingface_repo_name()
    chute_name = get_chute_name(hf_revision=revision)

    path_template = settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_MAIN
    template = Template(path_template.read_text())
    rendered = template.render(
        predict_endpoint=settings.CHUTES_MINER_PREDICT_ENDPOINT,
        repo_name=hf_repo_name,
        revision=revision,
        chute_user=settings.CHUTES_USERNAME,
        chute_name=chute_name,
        schema_defs=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_SCHEMAS
        ).read_text(),
        setup_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_SETUP_UTILS
        ).read_text(),
        load_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_LOAD_UTILS
        ).read_text(),
        predict_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_PREDICT_UTILS
        ).read_text(),
    )
    return rendered


async def get_chute_slug_and_id(revision: str) -> tuple[str, str | None]:
    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "chutes",
        "get",
        get_chute_name(hf_revision=revision),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
    )
    out, _ = await proc.communicate()
    log = out.decode(errors="ignore")
    logger.info(log[-800:])
    if proc.returncode != 0:
        logger.error(log)
        raise ValueError("Chutes Query failed.")
    json_tail_of_log = "{" + "{".join(log.split("{")[1:])
    logger.info(json_tail_of_log)
    try:
        json_response = loads(json_tail_of_log)
    except JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from chutes output: {e}")
        json_response = {}
    slug = json_response.get("slug")
    chute_id = json_response.get("chute_id")
    if slug:
        logger.info(f"Slug found: {slug}\n Chute Id: {chute_id}")
        return slug, chute_id
    slug = guess_chute_slug(hf_revision=revision)
    logger.info(f"No Slug returned. Guessing Slug {slug}\n Chute Id: {chute_id}")
    return slug, chute_id


async def build_chute(path: Path) -> None:
    logger.info(
        "🚧 Building model on chutes... This may take a while. Please don't exit."
    )

    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "build",
        f"{path.stem}:chute",
        "--public",
        "--wait",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
        cwd=str(path.parent),
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    # Read and log output line by line as it appears
    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes build] {decoded_line}")

    # out, _ = await proc.communicate()
    # log = out.decode(errors="ignore")
    # logger.info(log[-800:])
    # if proc.returncode != 0:
    returncode = await proc.wait()
    if returncode != 0:
        # logger.error(log)
        raise ValueError("Chutes building failed.")


async def deploy_chute(path: Path) -> None:
    logger.info("🚀 Deploying model to chutes... This may take a moment..")

    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "deploy",
        f"{path.stem}:chute",
        "--public",
        "--logging-enabled",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
        cwd=str(path.parent),
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes deploy] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        raise ValueError("Chutes deployment failed.")


async def delete_chute(revision: str) -> None:
    logger.info(" Removing model from chutes..")

    settings = get_settings()
    chute_slug, _ = await get_chute_slug_and_id(revision=revision)
    proc = await create_subprocess_exec(
        "chutes",
        "chutes",
        "delete",
        chute_slug,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes delete] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        raise ValueError("Chutes delete failed.")


async def build_and_deploy_chute(path: Path) -> None:
    settings = get_settings()
    if not settings.CHUTES_API_KEY.get_secret_value():
        raise ValueError("CHUTES_API_KEY missing.")
    chmod(str(path), stat(str(path)).st_mode | S_IEXEC)
    await build_chute(path=path)
    await deploy_chute(path=path)


async def call_chutes_list_models() -> dict | str:
    settings = get_settings()
    session = await get_async_client()
    async with session.get(
        f"{settings.CHUTES_MINERS_ENDPOINT}/chutes/",
        headers={"Authorization": settings.CHUTES_API_KEY.get_secret_value()},
    ) as response:
        t = await response.text()
        logger.info(t)
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {t[:200]}")
        try:
            return await response.json()
        except Exception as e:
            logger.error(e)
            return t


async def resolve_chute_id_and_slug(model_name: str) -> tuple[str, str]:
    """Jon: you can always query the /chutes/ endpoint (either list, or /chutes/{chute_id} on the API and extract the slug parameter for the subdomain"""
    chute_id = None
    chute_slug = None

    response = await call_chutes_list_models()
    if isinstance(response, dict) and "items" in response:
        chutes_list = response["items"]
    elif isinstance(response, list):
        chutes_list = response
    else:
        logger.error(response)
        chutes_list = []

    for ch in reversed(chutes_list):
        if any(ch.get(k) == model_name for k in ("model_name", "name", "readme")):
            chute_id = ch.get("chute_id") or ch.get("name") or ch.get("readme")
            chute_slug = ch.get("slug")
    if not chute_id or not chute_slug:
        raise Exception("Could not resolve chute_id/slug after deploy.")

    return chute_id, chute_slug


async def deploy_to_chutes(revision: str, skip: bool) -> tuple[str, str]:
    if skip:
        return None, None

    try:
        chute_deployment_script = render_chute_template(
            revision=revision,
        )
        with temporary_chutes_config_file(prefix="sv_chutes", delete=True) as (
            tmp,
            tmp_path,
        ):
            tmp.write(chute_deployment_script)
            tmp.flush()
            logger.info(f"Wrote Chute config: {tmp_path}")
            await build_and_deploy_chute(path=tmp_path)
            logger.info(f"Chute deployment successful")
        chute_slug, chute_id = await get_chute_slug_and_id(revision=revision)
        # NOTE: Alternative method below - but API doesnt seem to have as up-to-date information returned
        # chute_id, chute_slug = await resolve_chute_id_and_slug(model_name=model_name)
        logger.info(f"Deployed chute_id={chute_id} slug={chute_slug}")
        return chute_id, chute_slug
    except Exception as e:
        logger.error(e)
        return None, None
