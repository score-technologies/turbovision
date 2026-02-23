from pathlib import Path
import click

from scorevision.utils.settings import get_settings
from scorevision.utils.chutes_helpers import deploy_to_chutes

from scorevision.utils.huggingface_helpers import (
    create_update_or_verify_huggingface_repo,
)
from scorevision.utils.bittensor_helpers import on_chain_commit
from scorevision.utils.manifest import (
    get_current_manifest,
    load_manifest_from_public_index,
)


async def _resolve_element_id_from_manifest(
    element_id: str | None,
    *,
    skip_bittensor_commit: bool,
) -> str | None:
    if element_id:
        return element_id
    if skip_bittensor_commit:
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

    element_ids = [str(getattr(element, "id", "")).strip() for element in manifest.elements]
    element_ids = list(dict.fromkeys(eid for eid in element_ids if eid))
    if not element_ids:
        raise click.ClickException("No element IDs found in the current manifest.")

    click.echo("Available element IDs from manifest:")
    for idx, eid in enumerate(element_ids, start=1):
        click.echo(f"  {idx}. {eid}")

    choice = click.prompt(
        "Select element ID (number)",
        type=click.IntRange(1, len(element_ids)),
    )
    return element_ids[choice - 1]


async def push_ml_model(
    ml_model_path: Path | None,
    hf_revision: str | None,
    skip_chutes_deploy: bool,
    skip_bittensor_commit: bool,
    element_id: str | None,
) -> None:
    element_id = await _resolve_element_id_from_manifest(
        element_id,
        skip_bittensor_commit=skip_bittensor_commit,
    )

    hf_revision = await create_update_or_verify_huggingface_repo(
        model_path=ml_model_path, hf_revision=hf_revision
    )

    chute_id, chute_slug = await deploy_to_chutes(
        revision=hf_revision,
        skip=skip_chutes_deploy,
    )

    if chute_id:
        await on_chain_commit(
            skip=skip_bittensor_commit,
            revision=hf_revision,
            chute_id=chute_id,
            chute_slug=chute_slug,
            element_id=element_id,
        )
