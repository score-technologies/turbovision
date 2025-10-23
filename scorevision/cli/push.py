from pathlib import Path

from scorevision.utils.settings import get_settings
from scorevision.utils.chutes_helpers import deploy_to_chutes, share_chute

from scorevision.utils.huggingface_helpers import (
    create_update_or_verify_huggingface_repo,
)
from scorevision.utils.bittensor_helpers import on_chain_commit


async def push_ml_model(
    ml_model_path: Path | None,
    hf_revision: str | None,
    skip_chutes_deploy: bool,
    skip_bittensor_commit: bool,
) -> None:
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
        )
