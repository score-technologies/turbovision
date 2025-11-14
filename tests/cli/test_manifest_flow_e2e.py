from pathlib import Path
from click.testing import CliRunner
from scorevision.cli.manifest import manifest_cli

def test_manifest_full_flow(tmp_path: Path):
    runner = CliRunner()

    manifest_path = tmp_path / "full.yaml"
    key_path = Path("tests/fixtures/private_key_ed25519.pem")

    # 1. CREATE
    res1 = runner.invoke(
        manifest_cli,
        [
            "create",
            "--template", "default-football",
            "--window-id", "2025-10-24",
            "--expiry-block", "77000",
            "--output", str(manifest_path),
        ],
    )
    assert res1.exit_code == 0
    assert manifest_path.exists()

    # 2. VALIDATE (unsigned)
    res2 = runner.invoke(
        manifest_cli,
        ["validate", str(manifest_path)]
    )
    assert res2.exit_code == 0
    assert "unsigned" in res2.output or "valid" in res2.output

    # 3. PUBLISH (sign)
    res3 = runner.invoke(
        manifest_cli,
        [
            "publish",
            str(manifest_path),
            "--signing-key-path",
            str(key_path),
        ],
    )
    assert res3.exit_code == 0

    # 4. VALIDATE (signed)
    res4 = runner.invoke(
        manifest_cli,
        [
            "validate",
            str(manifest_path),
            "--public-key",
            key_path.read_bytes().hex(),  # hex is allowed
        ],
    )
    assert res4.exit_code == 0

