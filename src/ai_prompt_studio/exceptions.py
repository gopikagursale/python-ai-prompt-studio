"""Domain exceptions with messages safe to show in the CLI."""


class PromptStudioError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(PromptStudioError):
    """Environment configuration is missing or invalid."""


class PromptValidationError(PromptStudioError):
    """Prompt input does not satisfy a template."""


class ModelClientError(PromptStudioError):
    """A safe, user-facing model request failure."""


class AuthenticationFailure(ModelClientError):
    """The configured credentials were rejected."""


class RateLimitFailure(ModelClientError):
    """The provider refused the request because of a rate limit."""


class NetworkFailure(ModelClientError):
    """The provider could not be reached."""


class RequestTimeoutFailure(ModelClientError):
    """The provider request timed out."""


class StructuredOutputError(ModelClientError):
    """A structured-output request did not yield a valid typed result."""


class StructuredRefusalError(StructuredOutputError):
    """The model explicitly refused the structured-output request."""


class IncompleteStructuredOutputError(StructuredOutputError):
    """The API reported an incomplete or truncated response."""


class MissingParsedOutputError(StructuredOutputError):
    """A completed response did not contain parsed structured data."""


class UnsupportedStructuredOutputError(StructuredOutputError):
    """The configured model or selected schema is unsupported."""


class StructuredValidationFailure(StructuredOutputError):
    """Returned data failed local Pydantic validation."""
