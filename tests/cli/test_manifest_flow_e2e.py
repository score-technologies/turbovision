from pathlib import Path
from click.testing import CliRunner
from scorevision.cli.manifest import manifest_cli

def test_manifest_full_flow(tmp_path: Path, generated_ed25519_key: Path):
    """
    Full manifest flow:
    1. Create a manifest
    2. Validate unsigned
    3. Publish (sign) with a dynamically generated Ed25519 key
    4. Validate signed
    """
    runner = CliRunner()

    manifest_path = tmp_path / "full.yaml"

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

    # 3. PUBLISH (sign) with fixture key
    res3 = runner.invoke(
        manifest_cli,
        [
            "publish",
            str(manifest_path),
            "--signing-key-path",
            str(generated_ed25519_key),
        ],
    )
    assert res3.exit_code == 0
    assert "Signed manifest" in res3.output

    # 4. VALIDATE (signed)
    pub_key_bytes = generated_ed25519_key.read_bytes()
    res4 = runner.invoke(
        manifest_cli,
        [
            "validate",
            str(manifest_path),
            "--public-key",
            pub_key_bytes.hex(),  # hex representation allowed
        ],
    )
    assert res4.exit_code == 0

