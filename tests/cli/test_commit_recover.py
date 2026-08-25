from unittest.mock import AsyncMock

from click.testing import CliRunner

from scorevision import app
from scorevision.cli import open_source_miner
from scorevision.utils import bittensor_helpers


def test_commit_recover_submits_selected_element(monkeypatch):
    resolve = AsyncMock(return_value="element-a")
    commit = AsyncMock(return_value=True)
    monkeypatch.setattr(open_source_miner, "_resolve_element_id_from_manifest", resolve)
    monkeypatch.setattr(bittensor_helpers, "on_chain_commit_recover", commit)

    result = CliRunner().invoke(
        app,
        ["commit_recover", "--element-id", "element-a"],
    )

    assert result.exit_code == 0
    commit.assert_awaited_once_with(element_id="element-a", skip=False)
    assert "Commit recovery submitted" in result.output
