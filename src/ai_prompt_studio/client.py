"""Small, testable adapter for the OpenAI Responses API."""

from __future__ import annotations

from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .config import AppSettings
from .exceptions import (
    AuthenticationFailure,
    ModelClientError,
    NetworkFailure,
    RateLimitFailure,
    RequestTimeoutFailure,
)
from .models import GenerationResult, RenderedPrompt, UsageMetadata


class ResponsesResource(Protocol):
    """The subset of the SDK response resource used by this application."""

    def create(self, **kwargs: Any) -> Any: ...


class SDKClient(Protocol):
    responses: ResponsesResource


class OpenAIModelClient:
    """Execute stateless prompts through the official Responses API."""

    def __init__(self, settings: AppSettings, sdk_client: SDKClient | None = None) -> None:
        self.model = settings.model
        self._client = sdk_client or OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    def generate(self, prompt: RenderedPrompt) -> GenerationResult:
        """Send one prompt and normalize text and usage metadata."""

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=prompt.developer_instructions,
                input=prompt.user_prompt,
                store=False,
            )
        except AuthenticationError:
            raise AuthenticationFailure(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from None
        except RateLimitError:
            raise RateLimitFailure(
                "OpenAI rate limit reached. Wait and try again, or check account limits."
            ) from None
        except APITimeoutError:
            raise RequestTimeoutFailure("The OpenAI request timed out. Try again later.") from None
        except APIConnectionError:
            raise NetworkFailure(
                "Could not connect to OpenAI. Check the network and try again."
            ) from None
        except APIError:
            raise ModelClientError("The OpenAI request failed. Try again later.") from None
        except Exception:
            # Do not echo provider exceptions: request details can contain secrets.
            raise ModelClientError(
                "An unexpected error occurred while contacting OpenAI."
            ) from None

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ModelClientError("OpenAI returned no response text.")

        return GenerationResult(text=text, usage=self._extract_usage(response))

    @staticmethod
    def _extract_usage(response: Any) -> UsageMetadata | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        try:
            return UsageMetadata(
                input_tokens=int(usage.input_tokens),
                output_tokens=int(usage.output_tokens),
                total_tokens=int(usage.total_tokens),
            )
        except (AttributeError, TypeError, ValueError):
            return None
