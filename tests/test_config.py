from __future__ import annotations

import pytest

from ai_prompt_studio.config import AppSettings
from ai_prompt_studio.exceptions import ConfigurationError


def test_model_is_selected_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-environment")
    monkeypatch.setenv("OPENAI_MODEL", "model-from-environment")

    settings = AppSettings.from_env()

    assert settings.model == "model-from-environment"


def test_configuration_error_does_not_expose_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "extremely-secret-api-key"
    monkeypatch.setattr("ai_prompt_studio.config.load_dotenv", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ConfigurationError) as caught:
        AppSettings.from_env()

    assert secret not in str(caught.value)
