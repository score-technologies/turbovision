# tests/cli/test_manifest_publish.py

import os
from click.testing import CliRunner
from types import SimpleNamespace

import pytest
from scorevision.cli.manifest import manifest_cli


def test_publish_updates_index(tmp_path, signed_manifest_file, signing_key_hex, fake_settings, r2_mock_store):
    store, mock_get, mock_put, mock_delete = r2_mock_store

    # Inject the key via environment variable instead of PEM path
    os.environ["TEE_KEY_HEX"] = signing_key_hex

    from unittest.mock import patch

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
            ],
        )

        assert result.exit_code == 0, result.output

