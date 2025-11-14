import json
from unittest.mock import patch
from click.testing import CliRunner
from types import SimpleNamespace

from scorevision.cli.manifest import manifest_cli
from scorevision.utils.manifest import Manifest


def test_publish_updates_index(tmp_path, signed_manifest_file, generated_ed25519_key, fake_settings, r2_mock_store):
    store, mock_get, mock_put, mock_delete = r2_mock_store

    from unittest.mock import patch
    from click.testing import CliRunner
    from scorevision.cli.manifest import manifest_cli

    with patch("scorevision.utils.r2.r2_get_object", side_effect=mock_get), \
         patch("scorevision.utils.r2.r2_put_json", side_effect=mock_put), \
         patch("scorevision.utils.r2.r2_delete_object", side_effect=mock_delete), \
         patch("scorevision.cli.manifest.get_settings", return_value=fake_settings):

        runner = CliRunner()
        result = runner.invoke(
            manifest_cli,
            [
                "publish",
                str(signed_manifest_file),
                "--signing-key-path",
                str(generated_ed25519_key),
            ],
        )

        assert result.exit_code == 0

