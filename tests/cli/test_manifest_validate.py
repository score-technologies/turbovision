from click.testing import CliRunner
from pathlib import Path
from scorevision.cli.manifest import manifest_cli

SAMPLE_MANIFEST = """
window_id: "2025-10-24"
version: "1.3"
expiry_block: 123456
tee:
  trusted_share_gamma: 0.2
elements: []
"""

def test_manifest_validate_basic(tmp_path: Path):
    runner = CliRunner()

    path = tmp_path / "manifest.yaml"
    path.write_text(SAMPLE_MANIFEST)

    result = runner.invoke(manifest_cli, ["validate", str(path)])
    assert result.exit_code == 0
    assert "schema looks valid" in result.output or "successful" in result.output

