from pathlib import Path

from unittest.mock import patch
from click.testing import CliRunner

from scorevision.cli.manifest import manifest_cli


def test_manifest_list_empty_index(tmp_path: Path, fake_settings):
    """List manifests when no index exists."""
    runner = CliRunner()
    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch("scorevision.cli.manifest.r2_get_object", return_value=(None, None)):
            result = runner.invoke(manifest_cli, ["list"])
            assert result.exit_code == 0
            assert "No manifests published" in result.output


def test_manifest_list_with_windows(tmp_path: Path, fake_settings, fake_index_bytes):
    """List manifests with some windows in index."""
    runner = CliRunner()
    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch(
            "scorevision.cli.manifest.r2_get_object",
            return_value=(fake_index_bytes, None),
        ):
            result = runner.invoke(manifest_cli, ["list"])
            assert result.exit_code == 0
            assert "Published manifests" in result.output
            assert "Window: 2025-10-24" in result.output
            assert "Window: 2025-10-25" in result.output
