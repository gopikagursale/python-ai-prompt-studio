"""Validated prompt construction."""

from __future__ import annotations

from collections.abc import Mapping

from .exceptions import PromptValidationError
from .models import PromptTemplate, RenderedPrompt


class PromptBuilder:
    """Render user variables and assemble higher-priority instructions."""

    def __init__(self, developer_instructions: str) -> None:
        cleaned = developer_instructions.strip()
        if not cleaned:
            raise PromptValidationError("Developer instructions cannot be empty.")
        self._base_instructions = cleaned

    def build(self, template: PromptTemplate, variables: Mapping[str, str]) -> RenderedPrompt:
        """Validate variables and produce separated request content."""

        required = set(template.required_input_variables)
        provided = set(variables)
        missing = sorted(required - provided)
        extra = sorted(provided - required)
        if missing:
            raise PromptValidationError(f"Missing required variables: {', '.join(missing)}.")
        if extra:
            raise PromptValidationError(f"Unexpected variables: {', '.join(extra)}.")

        cleaned_values: dict[str, str] = {}
        for name in template.required_input_variables:
            value = variables[name]
            if not isinstance(value, str):
                raise PromptValidationError(f"Variable '{name}' must be text.")
            if not value.strip():
                raise PromptValidationError(f"Variable '{name}' cannot be empty.")
            # Preserve meaningful whitespace and all braces/special characters.
            cleaned_values[name] = value

        user_prompt = template.user_prompt_template.format_map(cleaned_values)

        sections = [
            self._base_instructions,
            f"Task: {template.task_description}",
            "Template instructions:\n- " + "\n- ".join(template.instructions),
        ]
        if template.output_constraints:
            sections.append("Output constraints:\n- " + "\n- ".join(template.output_constraints))

        return RenderedPrompt(
            template_id=template.template_id,
            template_name=template.name,
            template_version=template.version,
            developer_instructions="\n\n".join(sections),
            user_prompt=user_prompt,
        )
