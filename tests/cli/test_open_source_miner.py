from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from scorevision.cli import open_source_miner


@pytest.mark.asyncio
async def test_repo_is_published_only_after_successful_on_chain_commit(monkeypatch):
    events = Mock()

    async def prepare_repo(*, model_path, hf_revision):
        events("hf_private")
        return "revision-1"

    async def deploy(*, revision, skip):
        events("chutes_ready")
        return "chute-1", "slug-1"

    async def commit(**kwargs):
        events("on_chain_committed")
        return True

    monkeypatch.setattr(
        open_source_miner,
        "create_update_or_verify_huggingface_repo",
        prepare_repo,
    )
    monkeypatch.setattr(open_source_miner, "deploy_to_chutes", deploy)
    monkeypatch.setattr(open_source_miner, "on_chain_commit", commit)
    monkeypatch.setattr(
        open_source_miner,
        "set_huggingface_repo_visibility",
        lambda *, private: events("hf_public", private),
    )

    await open_source_miner.deploy_miner(
        ml_model_path=Path("model"),
        hf_revision=None,
        skip_chutes_deploy=False,
        skip_bittensor_commit=False,
        element_id="element-1",
    )

    assert events.call_args_list == [
        call("hf_private"),
        call("chutes_ready"),
        call("on_chain_committed"),
        call("hf_public", False),
    ]


@pytest.mark.asyncio
async def test_repo_stays_private_when_on_chain_commit_fails(monkeypatch):
    monkeypatch.setattr(
        open_source_miner,
        "create_update_or_verify_huggingface_repo",
        AsyncMock(return_value="revision-1"),
    )
    monkeypatch.setattr(
        open_source_miner,
        "deploy_to_chutes",
        AsyncMock(return_value=("chute-1", "slug-1")),
    )
    monkeypatch.setattr(
        open_source_miner, "on_chain_commit", AsyncMock(return_value=False)
    )
    publish = Mock()
    monkeypatch.setattr(open_source_miner, "set_huggingface_repo_visibility", publish)

    with pytest.raises(
        open_source_miner.click.ClickException,
        match="On-chain commitment failed; the Hugging Face repository remains private",
    ):
        await open_source_miner.deploy_miner(
            ml_model_path=None,
            hf_revision="revision-1",
            skip_chutes_deploy=False,
            skip_bittensor_commit=False,
            element_id="element-1",
        )

    publish.assert_not_called()


@pytest.mark.asyncio
async def test_no_commit_keeps_repo_private_without_failing(monkeypatch):
    monkeypatch.setattr(
        open_source_miner,
        "create_update_or_verify_huggingface_repo",
        AsyncMock(return_value="revision-1"),
    )
    monkeypatch.setattr(
        open_source_miner,
        "deploy_to_chutes",
        AsyncMock(return_value=("chute-1", "slug-1")),
    )
    monkeypatch.setattr(
        open_source_miner, "on_chain_commit", AsyncMock(return_value=False)
    )
    publish = Mock()
    monkeypatch.setattr(open_source_miner, "set_huggingface_repo_visibility", publish)

    await open_source_miner.deploy_miner(
        ml_model_path=None,
        hf_revision="revision-1",
        skip_chutes_deploy=False,
        skip_bittensor_commit=True,
        element_id="element-1",
    )

    publish.assert_not_called()
