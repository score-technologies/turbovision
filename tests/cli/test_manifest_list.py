from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from scorevision.cli.manifest import manifest_cli
import json


def test_manifest_list_empty_index(tmp_path: Path):
    """List manifests when no index exists."""
    runner = CliRunner()
    fake_settings = MagicMock()
    fake_settings.SCOREVISION_BUCKET = "fake-bucket"

    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch("scorevision.cli.manifest.r2_get_object", return_value=(None, None)):
            result = runner.invoke(manifest_cli, ["list"])
            assert result.exit_code == 0
            assert "No manifests published" in result.output


def test_manifest_list_with_windows(tmp_path: Path):
    """List manifests with some windows in index."""
    runner = CliRunner()
    fake_settings = MagicMock()
    fake_settings.SCOREVISION_BUCKET = "fake-bucket"

    fake_index = {
        "windows": {
            "2025-10-24": {
                "current": "sha256:abc123",
                "version": "1.3",
                "expiry_block": 123456,
                "updated_at": "2025-10-01T12:00:00Z",
            },
            "2025-10-25": {
                "current": "sha256:def456",
                "version": "1.3",
                "expiry_block": 123460,
                "updated_at": "2025-10-02T12:00:00Z",
            },
        }
    }
    index_bytes = json.dumps(fake_index).encode("utf-8")

    with patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):
        with patch(
            "scorevision.cli.manifest.r2_get_object", return_value=(index_bytes, None)
        ):
            result = runner.invoke(manifest_cli, ["list"])
            assert result.exit_code == 0
            assert "Published manifests" in result.output
            assert "Window: 2025-10-24" in result.output
            assert "Window: 2025-10-25" in result.output
