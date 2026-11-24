"""Prompt and response data models."""

from __future__ import annotations

import re
from string import Formatter

from pydantic import BaseModel, ConfigDict, Field, model_validator

VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PromptTemplate(BaseModel):
    """A versioned, reusable prompt definition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    template_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    task_description: str = Field(min_length=1)
    instructions: tuple[str, ...] = Field(min_length=1)
    required_input_variables: tuple[str, ...] = Field(min_length=1)
    output_constraints: tuple[str, ...] = ()
    version: str = Field(min_length=1, pattern=r"^\d+\.\d+\.\d+$")
    user_prompt_template: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_placeholders(self) -> "PromptTemplate":
        required = self.required_input_variables
        if len(set(required)) != len(required):
            raise ValueError("required input variables must be unique")
        if any(not VARIABLE_NAME.fullmatch(name) for name in required):
            raise ValueError("input variable names must be simple identifiers")

        found: list[str] = []
        try:
            parsed = Formatter().parse(self.user_prompt_template)
            for _, field_name, format_spec, conversion in parsed:
                if field_name is None:
                    continue
                if not VARIABLE_NAME.fullmatch(field_name):
                    raise ValueError("template placeholders must be simple variable names")
                if format_spec or conversion:
                    raise ValueError("format specifications and conversions are not allowed")
                found.append(field_name)
        except ValueError as exc:
            raise ValueError(f"invalid user prompt template: {exc}") from exc

        if set(found) != set(required):
            raise ValueError("template placeholders must exactly match required input variables")
        return self


class RenderedPrompt(BaseModel):
    """Validated request content with instructions and input kept separate."""

    model_config = ConfigDict(frozen=True)

    template_id: str
    template_name: str
    template_version: str
    developer_instructions: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)


class UsageMetadata(BaseModel):
    """Token counts returned by the provider, when available."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class GenerationResult(BaseModel):
    """Text and optional metadata from a completed model response."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: UsageMetadata | None = None
