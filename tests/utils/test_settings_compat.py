from scorevision.utils.settings import get_settings


def test_legacy_r2_env_names_are_accepted(monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "legacy-bucket")
    monkeypatch.setenv("R2_BUCKET_PUBLIC_URL", "https://pub-legacy.r2.dev")
    monkeypatch.setenv("R2_ACCOUNT_ID", "legacy-account")
    monkeypatch.setenv("R2_WRITE_ACCESS_KEY_ID", "legacy-key")
    monkeypatch.setenv("R2_WRITE_SECRET_ACCESS_KEY", "legacy-secret")
    monkeypatch.setenv("SCOREVISION_RESULTS_PREFIX", "legacy-results")
    monkeypatch.setenv("SCOREVISION_R2_CONCURRENCY", "11")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.SCOREVISION_BUCKET == "legacy-bucket"
    assert settings.SCOREVISION_PUBLIC_RESULTS_URL == "https://pub-legacy.r2.dev"
    assert settings.CENTRAL_R2_ACCOUNT_ID.get_secret_value() == "legacy-account"
    assert settings.CENTRAL_R2_WRITE_ACCESS_KEY_ID.get_secret_value() == "legacy-key"
    assert (
        settings.CENTRAL_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
        == "legacy-secret"
    )
    assert settings.CENTRAL_R2_RESULTS_PREFIX == "legacy-results"
    assert settings.CENTRAL_R2_CONCURRENCY == 11
    get_settings.cache_clear()


def test_legacy_hf_env_names_are_accepted(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setenv("HF_USER", "legacy-user")
    monkeypatch.setenv("HF_TOKEN", "legacy-token")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.HUGGINGFACE_USERNAME == "legacy-user"
    assert settings.HUGGINGFACE_API_KEY.get_secret_value() == "legacy-token"
    get_settings.cache_clear()


def test_scorevision_public_results_url_env_name_is_accepted(monkeypatch):
    monkeypatch.delenv("R2_BUCKET_PUBLIC_URL", raising=False)
    monkeypatch.setenv("SCOREVISION_PUBLIC_RESULTS_URL", "https://pub-scorevision.r2.dev")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.SCOREVISION_PUBLIC_RESULTS_URL == "https://pub-scorevision.r2.dev"
    get_settings.cache_clear()
