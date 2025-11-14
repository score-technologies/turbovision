import json
from pathlib import Path
from click.testing import CliRunner
from scorevision.cli.manifest import manifest_cli

def test_manifest_create(tmp_path: Path):
    runner = CliRunner()

    out = tmp_path / "test.yaml"

    result = runner.invoke(
        manifest_cli,
        [
            "create",
            "--template", "default-football",
            "--window-id", "2025-10-24",
            "--expiry-block", "123456",
            "--output", str(out),
        ],
    )

    assert result.exit_code == 0
    assert out.exists()

    data = out.read_text()
    assert "window_id: 2025-10-24" in data
    assert "expiry_block: 123456" in data
    assert "tee:" in data
    assert "trusted_share_gamma" in data
