# tests/cli/test_manifest_create.py

import os
from pathlib import Path
from click.testing import CliRunner
import pytest

from scorevision.cli.manifest import manifest_cli


def test_manifest_create(tmp_path: Path, signing_key_hex: str):
    """Test manifest creation using a key from environment variable."""
    # Set the environment variable so CLI picks it up
    os.environ["TEE_KEY_HEX"] = signing_key_hex

    runner = CliRunner()
    out = tmp_path / "test.yaml"

    result = runner.invoke(
        manifest_cli,
        [
            "create",
            "--template",
            "default-football",
            "--window-id",
            "2025-10-24",
            "--expiry-block",
            "123456",
            "--output",
            str(out),
        ],
    )

    # Assertions
    assert result.exit_code == 0, result.output
    assert out.exists(), "Manifest file was not created"

    data = out.read_text()
    assert "window_id" in data
    assert "2025-10-24" in data
    assert "expiry_block: 123456" in data
    assert "tee:" in data
    assert "trusted_share_gamma" in data
