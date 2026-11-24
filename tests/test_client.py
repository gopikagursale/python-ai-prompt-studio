from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from ai_prompt_studio.client import OpenAIModelClient
from ai_prompt_studio.exceptions import (
    AuthenticationFailure,
    ModelClientError,
    NetworkFailure,
    RateLimitFailure,
    RequestTimeoutFailure,
)
from ai_prompt_studio.models import RenderedPrompt


@pytest.fixture
def prompt() -> RenderedPrompt:
    return RenderedPrompt(
        template_id="test",
        template_name="Test",
        template_version="1.0.0",
        developer_instructions="Trusted developer instructions.",
        user_prompt="Untrusted user content.",
    )


def _sdk_with_response(response: object) -> Mock:
    sdk = Mock()
    sdk.responses.create.return_value = response
    return sdk


def test_sdk_receives_timeout_retry_and_key_settings(settings, monkeypatch) -> None:
    sdk = Mock()
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr("ai_prompt_studio.client.OpenAI", constructor)

    OpenAIModelClient(settings)

    constructor.assert_called_once_with(api_key="test-secret-key", timeout=12.0, max_retries=1)


def test_correct_model_request_arguments_and_response_extraction(
    settings, prompt: RenderedPrompt
) -> None:
    usage = SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14)
    sdk = _sdk_with_response(SimpleNamespace(output_text="Answer", usage=usage))

    result = OpenAIModelClient(settings, sdk).generate(prompt)

    sdk.responses.create.assert_called_once_with(
        model="test-model",
        instructions="Trusted developer instructions.",
        input="Untrusted user content.",
        store=False,
    )
    assert result.text == "Answer"
    assert result.usage is not None
    assert result.usage.total_tokens == 14


def test_missing_usage_metadata(settings, prompt: RenderedPrompt) -> None:
    sdk = _sdk_with_response(SimpleNamespace(output_text="Answer", usage=None))

    result = OpenAIModelClient(settings, sdk).generate(prompt)

    assert result.usage is None


def test_authentication_error_is_safe(settings, prompt: RenderedPrompt) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = AuthenticationError(
        "provider leaked test-secret-key", response=Mock(status_code=401), body=None
    )

    with pytest.raises(AuthenticationFailure) as caught:
        OpenAIModelClient(settings, sdk).generate(prompt)

    assert "test-secret-key" not in str(caught.value)
    assert "authentication failed" in str(caught.value).lower()


def test_rate_limit_error_is_safe(settings, prompt: RenderedPrompt) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = RateLimitError(
        "provider leaked test-secret-key", response=Mock(status_code=429), body=None
    )

    with pytest.raises(RateLimitFailure) as caught:
        OpenAIModelClient(settings, sdk).generate(prompt)

    assert "test-secret-key" not in str(caught.value)
    assert "rate limit" in str(caught.value).lower()


def test_network_error_is_safe(settings, prompt: RenderedPrompt) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = APIConnectionError(
        message="provider leaked test-secret-key", request=Mock()
    )

    with pytest.raises(NetworkFailure) as caught:
        OpenAIModelClient(settings, sdk).generate(prompt)

    assert "test-secret-key" not in str(caught.value)
    assert "connect" in str(caught.value).lower()


def test_timeout_error_is_safe(settings, prompt: RenderedPrompt) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = APITimeoutError(request=Mock())

    with pytest.raises(RequestTimeoutFailure) as caught:
        OpenAIModelClient(settings, sdk).generate(prompt)

    assert "test-secret-key" not in str(caught.value)
    assert "timed out" in str(caught.value).lower()


def test_unexpected_error_does_not_expose_secrets(settings, prompt: RenderedPrompt) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = RuntimeError("test-secret-key")

    with pytest.raises(ModelClientError) as caught:
        OpenAIModelClient(settings, sdk).generate(prompt)

    assert "test-secret-key" not in str(caught.value)


def test_empty_response_text_is_rejected(settings, prompt: RenderedPrompt) -> None:
    sdk = _sdk_with_response(SimpleNamespace(output_text=" ", usage=None))

    with pytest.raises(ModelClientError, match="no response text"):
        OpenAIModelClient(settings, sdk).generate(prompt)
