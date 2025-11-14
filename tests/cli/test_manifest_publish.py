from pathlib import Path
from click.testing import CliRunner
from scorevision.cli.manifest import manifest_cli

SAMPLE_MANIFEST = """
window_id: "2025-10-24"
version: "1.3"
expiry_block: 123456
tee:
  trusted_share_gamma: 0.2
elements: []
"""

def test_manifest_publish(tmp_path: Path, generated_ed25519_key: Path):
    runner = CliRunner()

    mf_path = tmp_path / "manifest.yaml"
    mf_path.write_text(SAMPLE_MANIFEST)

    result = runner.invoke(
        manifest_cli,
        [
            "publish",
            str(mf_path),
            "--signing-key-path",
            str(generated_ed25519_key),
        ],
    )

    assert result.exit_code == 0
    assert "Signed manifest" in result.output

    # Confirm signature added
    contents = mf_path.read_text()
    assert "signature:" in contents

