from os import environ
from pathlib import Path

from click.testing import CliRunner
from unittest.mock import patch

from scorevision.cli.manifest import manifest_cli


def test_manifest_full_flow(
    tmp_path: Path, generated_ed25519_key: Path, fake_settings, r2_mock_store
) -> None:
    """
    Full manifest lifecycle:
    1. Create a manifest
    2. Validate unsigned
    3. Publish (sign + upload)
    4. Validate signed
    """
    store, mock_get, mock_put, mock_delete = r2_mock_store
    runner = CliRunner()
    manifest_path = tmp_path / "full.yaml"
    environ["TEE_KEY_HEX"] = generated_ed25519_key.read_text().strip()

    # ---------------------------
    # 1. CREATE
    # ---------------------------
    res1 = runner.invoke(
        manifest_cli,
        [
            "create",
            "--template",
            "default-football",
            "--window-id",
            "2025-10-24",
            "--expiry-block",
            "77000",
            "--output",
            str(manifest_path),
        ],
    )
    assert res1.exit_code == 0, res1.output
    assert manifest_path.exists(), f"Manifest file was not created: {manifest_path}"

    # ---------------------------
    # 2. VALIDATE (unsigned)
    # ---------------------------
    res2 = runner.invoke(manifest_cli, ["validate", str(manifest_path)])
    assert res2.exit_code == 0, res2.output

    # ---------------------------
    # 3. PUBLISH (sign + upload)
    # ---------------------------
    with (
        patch("scorevision.cli.manifest.r2_get_object", side_effect=mock_get),
        patch("scorevision.cli.manifest.r2_put_json", side_effect=mock_put),
        patch("scorevision.cli.manifest.r2_delete_object", side_effect=mock_delete),
        patch("scorevision.cli.manifest.get_settings", return_value=fake_settings),
    ):
        res3 = runner.invoke(
            manifest_cli,
            [
                "publish",
                str(manifest_path),
            ],
        )

    assert res3.exit_code == 0, res3.output

    # ---------------------------
    # 4. VALIDATE (signed)
    # ---------------------------
    res4 = runner.invoke(manifest_cli, ["validate", str(manifest_path)])
    assert res4.exit_code == 0, res4.output

