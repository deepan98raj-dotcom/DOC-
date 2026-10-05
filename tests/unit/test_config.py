from app.config import Settings


def test_default_settings() -> None:
    settings = Settings()

    assert settings.app_name == "DocMind AI Studio"
    assert settings.app_env == "development"
    assert settings.api_port == 8000
