"""Testable workflow helpers shared by the Streamlit presentation layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any

from .builder import PromptBuilder
from .client import OpenAIModelClient
from .exceptions import PromptValidationError
from .models import GenerationResult, RenderedPrompt
from .templates import get_template

PREVIEW_KEY = "studio.preview"
CONFIRMED_KEY = "studio.confirmed_prompt"
EXECUTION_KEY = "studio.execution"
EDITOR_KEY = "studio.prompt_editor"
TEMPLATE_WIDGET_KEY = "studio.selected_template"
STATUS_KEY = "studio.status"


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Metadata for the latest explicit model execution."""

    prompt: RenderedPrompt
    result: GenerationResult
    model: str
    latency_seconds: float
    status: str = "Success"


def variable_widget_key(template_id: str, variable: str) -> str:
    """Return a stable key scoped to one template and variable."""

    return f"studio.variable.{template_id}.{variable}"


def build_preview(
    builder: PromptBuilder, template_id: str, variables: Mapping[str, str]
) -> RenderedPrompt:
    """Validate a registry selection and render locally without an API call."""

    try:
        template = get_template(template_id)
    except KeyError:
        raise PromptValidationError("The selected prompt template is invalid.") from None
    return builder.build(template, variables)


def confirm_user_prompt(preview: RenderedPrompt, edited_user_prompt: str) -> RenderedPrompt:
    """Confirm the exact visible user prompt while preserving trusted instructions."""

    if not edited_user_prompt.strip():
        raise PromptValidationError("The final user prompt cannot be empty.")
    return preview.model_copy(update={"user_prompt": edited_user_prompt})


def prompt_fingerprint(prompt: RenderedPrompt) -> str:
    """Create a non-secret identity for UI consistency checks."""

    value = "\x1f".join(
        (
            prompt.template_id,
            prompt.template_version,
            prompt.developer_instructions,
            prompt.user_prompt,
        )
    )
    return sha256(value.encode("utf-8")).hexdigest()


def execute_confirmed_prompt(
    client: OpenAIModelClient,
    prompt: RenderedPrompt,
    model: str,
    clock: Callable[[], float] = perf_counter,
) -> ExecutionRecord:
    """Execute once and measure elapsed monotonic time around the API boundary."""

    started = clock()
    result = client.generate(prompt)
    elapsed = max(0.0, clock() - started)
    return ExecutionRecord(
        prompt=prompt,
        result=result,
        model=model,
        latency_seconds=elapsed,
    )


def clear_response_state(state: MutableMapping[str, Any]) -> None:
    """Remove only response-related state."""

    state.pop(EXECUTION_KEY, None)
    state.pop(STATUS_KEY, None)


def invalidate_preview_state(state: MutableMapping[str, Any]) -> None:
    """Remove preview, confirmation, editor, and any now-stale response."""

    state.pop(PREVIEW_KEY, None)
    state.pop(CONFIRMED_KEY, None)
    state.pop(EDITOR_KEY, None)
    clear_response_state(state)


def reset_prompt_state(state: MutableMapping[str, Any], template_id: str) -> None:
    """Restore the selected template form to empty defaults."""

    try:
        template = get_template(template_id)
    except KeyError:
        invalidate_preview_state(state)
        return
    for variable in template.required_input_variables:
        state.pop(variable_widget_key(template_id, variable), None)
    invalidate_preview_state(state)


def prompt_was_modified(preview: RenderedPrompt, confirmed: RenderedPrompt) -> bool:
    """Report whether confirmation changed the rendered user content."""

    return preview.user_prompt != confirmed.user_prompt
