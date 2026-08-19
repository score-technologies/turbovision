from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from scorevision.utils import chutes_helpers


@pytest.mark.asyncio
async def test_deploy_does_not_warm_up_or_health_check(monkeypatch):
    @contextmanager
    def temporary_config(**_kwargs):
        yield StringIO(), Path("chute.py")

    settings = SimpleNamespace(CHUTES_HF_TOKEN=SecretStr("hf-token"))
    build_and_deploy = AsyncMock()
    create_secret = AsyncMock()
    warmup = AsyncMock()
    health_check = AsyncMock()

    monkeypatch.setattr(chutes_helpers, "get_settings", lambda: settings)
    monkeypatch.setattr(
        chutes_helpers, "render_chute_template", lambda **_kwargs: "chute code"
    )
    monkeypatch.setattr(
        chutes_helpers, "temporary_chutes_config_file", temporary_config
    )
    monkeypatch.setattr(chutes_helpers, "build_and_deploy_chute", build_and_deploy)
    monkeypatch.setattr(
        chutes_helpers,
        "get_chute_slug_and_id",
        AsyncMock(return_value=("slug-1", "chute-1")),
    )
    monkeypatch.setattr(chutes_helpers, "create_huggingface_secret", create_secret)
    monkeypatch.setattr(chutes_helpers, "warmup_chute", warmup)
    monkeypatch.setattr(chutes_helpers, "verify_chute_health", health_check)

    result = await chutes_helpers.deploy_to_chutes("revision-1", skip=False)

    assert result == ("chute-1", "slug-1")
    build_and_deploy.assert_awaited_once_with(path=Path("chute.py"))
    create_secret.assert_awaited_once_with(chute_id="chute-1", token="hf-token")
    warmup.assert_not_awaited()
    health_check.assert_not_awaited()
