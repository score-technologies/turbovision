from os import environ
from types import SimpleNamespace

from click.testing import CliRunner
from unittest.mock import patch

from scorevision.cli.manifest import manifest_cli


def test_publish_updates_index(
    tmp_path, signed_manifest_file, signing_key_hex, fake_settings, r2_mock_store
) -> None:
    store, mock_get, mock_put, mock_delete = r2_mock_store

    environ["TEE_KEY_HEX"] = signing_key_hex

    with (
        patch("scorevision.cli.manifest.r2_get_object", side_effect=mock_get),
        patch("scorevision.cli.manifest.r2_put_json", side_effect=mock_put),
        patch("scorevision.cli.manifest.r2_delete_object", side_effect=mock_delete),
        patch("scorevision.cli.manifest.get_settings", return_value=fake_settings),
    ):
        runner = CliRunner()
        result = runner.invoke(
            manifest_cli,
            [
                "publish",
                str(signed_manifest_file),
            ],
        )
        assert result.exit_code == 0, result.output
