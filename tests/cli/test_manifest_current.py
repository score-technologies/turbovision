from pathlib import Path

from unittest.mock import patch
from click.testing import CliRunner

from scorevision.cli.manifest import manifest_cli


def test_current_no_index(tmp_path: Path, fake_settings):
    runner = CliRunner()

    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch("scorevision.cli.manifest.r2_get_object", return_value=(None, None)):
            result = runner.invoke(manifest_cli, ["current"])
            assert result.exit_code == 0
            assert "No manifests published" in result.output


def test_current_latest_manifest(tmp_path: Path, fake_settings, fake_index_bytes):
    runner = CliRunner()
    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch(
            "scorevision.cli.manifest.r2_get_object",
            return_value=(fake_index_bytes, None),
        ):
            result = runner.invoke(manifest_cli, ["current"])
            assert result.exit_code == 0
            assert "Latest manifest" in result.output
            assert "123460" in result.output
            assert "def456" in result.output


def test_current_active_for_block(tmp_path: Path, fake_settings, fake_index_bytes):
    """Return the active manifest for a given block."""
    runner = CliRunner()
    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch(
            "scorevision.cli.manifest.r2_get_object",
            return_value=(fake_index_bytes, None),
        ):
            # Block before first key's block number
            result = runner.invoke(manifest_cli, ["current", "--block", "123455"])
            assert result.exit_code == 0
            assert "No active manifest found" in result.output

            # Block equal to first key block
            result = runner.invoke(manifest_cli, ["current", "--block", "123456"])
            assert result.exit_code == 0
            assert "Active manifest for block 123456" in result.output

            # Block beyond all
            result = runner.invoke(manifest_cli, ["current", "--block", "1235000"])
            assert result.exit_code == 0
            assert "Active manifest for block 1235000" in result.output
