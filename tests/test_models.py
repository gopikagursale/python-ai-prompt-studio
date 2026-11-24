import pytest
from pydantic import ValidationError

from ai_prompt_studio.models import PromptTemplate


def _template(**overrides: object) -> PromptTemplate:
    data = {
        "template_id": "test-template",
        "name": "Test",
        "task_description": "Test a template.",
        "instructions": ("Be clear.",),
        "required_input_variables": ("value",),
        "version": "1.0.0",
        "user_prompt_template": "Value: {value}",
    }
    data.update(overrides)
    return PromptTemplate(**data)


def test_placeholder_must_match_declared_variables() -> None:
    with pytest.raises(ValidationError, match="exactly match"):
        _template(required_input_variables=("other",))


@pytest.mark.parametrize(
    "unsafe_template", ["{value.__class__}", "{value[0]}", "{value!r}", "{value:>10}"]
)
def test_unsafe_or_advanced_formatting_is_rejected(unsafe_template: str) -> None:
    with pytest.raises(ValidationError, match="invalid user prompt template"):
        _template(user_prompt_template=unsafe_template)
