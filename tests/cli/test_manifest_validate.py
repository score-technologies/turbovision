from pathlib import Path

from click.testing import CliRunner

from scorevision.cli.manifest import manifest_cli


def test_manifest_validate_basic(tmp_path: Path) -> None:
    runner = CliRunner()

    path = tmp_path / "manifest.yaml"
    path.write_text(
        """
window_id: "2025-10-24"
version: "1.3"
expiry_block: 123456
tee:
  trusted_share_gamma: 0.2
elements: []
"""
    )

    result = runner.invoke(manifest_cli, ["validate", str(path)])
    assert result.exit_code == 0
    assert "Schema is valid" in result.output
    assert "Manifest is unsigned" in result.output
    assert "Validation complete" in result.output
