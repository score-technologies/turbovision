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

def test_manifest_publish(tmp_path: Path):
    runner = CliRunner()

    mf_path = tmp_path / "manifest.yaml"
    key_path = Path("tests/fixtures/private_key_ed25519.pem")

    mf_path.write_text(SAMPLE_MANIFEST)

    result = runner.invoke(
        manifest_cli,
        [
            "publish",
            str(mf_path),
            "--signing-key-path",
            str(key_path),
        ],
    )

    assert result.exit_code == 0
    assert "Signed manifest" in result.output

    # Make sure signature added
    contents = mf_path.read_text()
    assert "signature:" in contents

