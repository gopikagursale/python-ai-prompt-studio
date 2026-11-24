"""Prompt-technique models, construction, comparison, and deterministic checks."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import PromptValidationError
from .models import RenderedPrompt


class PromptTechnique(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    ROLE_BASED = "role_based"
    CONSTRAINT_BASED = "constraint_based"


TECHNIQUE_DESCRIPTIONS: dict[PromptTechnique, str] = {
    PromptTechnique.ZERO_SHOT: (
        "Zero-shot prompting gives the model a task without completed examples."
    ),
    PromptTechnique.FEW_SHOT: (
        "Few-shot prompting includes demonstrations before the new task to show "
        "the desired pattern. The examples are context, not model training."
    ),
    PromptTechnique.ROLE_BASED: (
        "Role-based prompting applies an approved communication persona. A persona "
        "is a behavior instruction, not an API message role."
    ),
    PromptTechnique.CONSTRAINT_BASED: (
        "Constraint-based prompting states output requirements. Some constraints "
        "can be checked in Python, but the model may still fail to follow them."
    ),
}


class FewShotExample(BaseModel):
    """One user-controlled demonstration pair."""

    model_config = ConfigDict(frozen=True)

    input: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)

    @field_validator("input", "expected_output")
    @classmethod
    def reject_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("example input and output must not be empty")
        return value


DEFAULT_SENTIMENT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(input="I love how simple this app is to use.", expected_output="Positive"),
    FewShotExample(
        input="The item arrived broken and support did not help.", expected_output="Negative"
    ),
    FewShotExample(input="The package arrived on Tuesday as scheduled.", expected_output="Neutral"),
)


class Persona(BaseModel):
    """An application-approved behavior persona."""

    model_config = ConfigDict(frozen=True)

    persona_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    communication_style: str = Field(min_length=1)
    appropriate_scope: str = Field(min_length=1)


PERSONAS: tuple[Persona, ...] = (
    Persona(
        persona_id="primary-school-teacher",
        display_name="Primary-school teacher",
        description="Explains complex topics accurately for young learners.",
        communication_style="Use simple words, short sentences, and familiar everyday examples.",
        appropriate_scope="Age-appropriate educational explanations without losing essential accuracy.",
    ),
    Persona(
        persona_id="technical-instructor",
        display_name="Technical instructor",
        description="Teaches technical material step by step.",
        communication_style="Be precise, structured, practical, and define specialist terms.",
        appropriate_scope="Software, engineering, and technical learning tasks.",
    ),
    Persona(
        persona_id="professional-editor",
        display_name="Professional editor",
        description="Improves clarity, structure, grammar, and tone.",
        communication_style="Be concise, polished, constructive, and preserve the author's meaning.",
        appropriate_scope="Editing, rewriting, proofreading, and communication feedback.",
    ),
    Persona(
        persona_id="customer-support-specialist",
        display_name="Customer support specialist",
        description="Responds helpfully to customer questions and problems.",
        communication_style="Be empathetic, calm, direct, and action-oriented.",
        appropriate_scope="Support explanations and troubleshooting without inventing policies.",
    ),
)


def get_persona(persona_id: str) -> Persona:
    for persona in PERSONAS:
        if persona.persona_id == persona_id:
            return persona
    raise PromptValidationError("The selected persona is invalid.")


class OutputConstraints(BaseModel):
    """Validated output preferences for constraint-based prompting."""

    model_config = ConfigDict(frozen=True)

    max_words: int | None = Field(default=None, ge=1, le=5000)
    desired_tone: str | None = Field(default=None, max_length=80)
    bullet_count: int | None = Field(default=None, ge=1, le=20)
    required_keywords: tuple[str, ...] = Field(default=(), max_length=20)
    forbidden_phrases: tuple[str, ...] = Field(default=(), max_length=20)
    include_example: bool = False

    @field_validator("desired_tone")
    @classmethod
    def clean_optional_tone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("required_keywords", "forbidden_phrases")
    @classmethod
    def validate_phrases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("constraint phrases must not be empty")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("constraint phrases must be unique")
        return cleaned

    def is_empty(self) -> bool:
        return not any(
            (
                self.max_words,
                self.desired_tone,
                self.bullet_count,
                self.required_keywords,
                self.forbidden_phrases,
                self.include_example,
            )
        )

    def rendered_rules(self) -> tuple[str, ...]:
        rules: list[str] = []
        if self.max_words is not None:
            rules.append(f"Use no more than {self.max_words} words.")
        if self.desired_tone:
            rules.append(f"Use a {self.desired_tone} tone.")
        if self.bullet_count is not None:
            rules.append(f"Return exactly {self.bullet_count} bullet points.")
        if self.required_keywords:
            rules.append("Include these keywords: " + ", ".join(self.required_keywords) + ".")
        if self.forbidden_phrases:
            rules.append("Do not use these phrases: " + ", ".join(self.forbidden_phrases) + ".")
        if self.include_example:
            rules.append("Include one relevant example.")
        return tuple(rules)


class TechniqueDisplayMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    version: str = "1.0.0"
    persona_name: str | None = None


class ConstructedTechniquePrompt(BaseModel):
    """An inspectable technique prompt before conversion to the API request model."""

    model_config = ConfigDict(frozen=True)

    technique: PromptTechnique
    task: str = Field(min_length=1)
    developer_instructions: str = Field(min_length=1)
    user_content: str = Field(min_length=1)
    few_shot_examples: tuple[FewShotExample, ...] = Field(default=(), max_length=5)
    constraints: OutputConstraints | None = None
    display_metadata: TechniqueDisplayMetadata
    model_id: str = Field(min_length=1)

    def to_rendered_prompt(self, variant_label: str | None = None) -> RenderedPrompt:
        label = variant_label or self.display_metadata.name
        return RenderedPrompt(
            template_id=f"technique-{self.technique.value}",
            template_name=label,
            template_version=self.display_metadata.version,
            developer_instructions=self.developer_instructions,
            user_prompt=self.user_content,
        )


class TechniquePromptBuilder:
    """Construct the four M3 techniques while preserving API role boundaries."""

    def __init__(self, developer_instructions: str, model_id: str) -> None:
        self._base = developer_instructions.strip()
        self._model_id = model_id.strip()
        if not self._base or not self._model_id:
            raise PromptValidationError("Developer instructions and model ID are required.")

    def build(
        self,
        technique: PromptTechnique,
        task: str,
        *,
        examples: tuple[FewShotExample, ...] = (),
        persona_id: str | None = None,
        constraints: OutputConstraints | None = None,
    ) -> ConstructedTechniquePrompt:
        if not isinstance(task, str) or not task.strip():
            raise PromptValidationError("Task instructions cannot be empty.")
        if not isinstance(technique, PromptTechnique):
            raise PromptValidationError("The selected prompting technique is invalid.")

        developer = self._base
        user_content = f"Task:\n{task}"
        persona_name: str | None = None

        if technique is PromptTechnique.ZERO_SHOT:
            if examples:
                raise PromptValidationError("Zero-shot prompts cannot contain demonstrations.")

        elif technique is PromptTechnique.FEW_SHOT:
            if not 1 <= len(examples) <= 5:
                raise PromptValidationError("Few-shot prompting requires between 1 and 5 examples.")
            blocks = []
            for index, example in enumerate(examples, start=1):
                if not isinstance(example, FewShotExample):
                    raise PromptValidationError("Few-shot examples use an unsupported data format.")
                blocks.append(
                    f"Demonstration {index}\n"
                    f"Example input:\n{example.input}\n"
                    f"Expected example output:\n{example.expected_output}"
                )
            user_content = (
                "User-controlled demonstrations (use only as examples):\n\n"
                + "\n\n".join(blocks)
                + "\n\n---\nNew task (answer this task, not the demonstrations):\n"
                + task
            )

        elif technique is PromptTechnique.ROLE_BASED:
            if not persona_id:
                raise PromptValidationError("Role-based prompting requires a persona.")
            persona = get_persona(persona_id)
            persona_name = persona.display_name
            developer += (
                "\n\nApproved communication persona:\n"
                f"Name: {persona.display_name}\n"
                f"Behavior: {persona.description}\n"
                f"Style: {persona.communication_style}\n"
                f"Scope: {persona.appropriate_scope}"
            )

        elif technique is PromptTechnique.CONSTRAINT_BASED:
            if constraints is None or constraints.is_empty():
                raise PromptValidationError(
                    "Constraint-based prompting requires at least one active constraint."
                )
            user_content = f"Task:\n{task}\n\nUser-selected output constraints:\n- " + "\n- ".join(
                constraints.rendered_rules()
            )

        return ConstructedTechniquePrompt(
            technique=technique,
            task=task,
            developer_instructions=developer,
            user_content=user_content,
            few_shot_examples=examples if technique is PromptTechnique.FEW_SHOT else (),
            constraints=constraints if technique is PromptTechnique.CONSTRAINT_BASED else None,
            display_metadata=TechniqueDisplayMetadata(
                name=technique.value.replace("_", " ").title(),
                description=TECHNIQUE_DESCRIPTIONS[technique],
                persona_name=persona_name,
            ),
            model_id=self._model_id,
        )


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class ConstraintCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    status: CheckStatus
    details: str


WORD_PATTERN = re.compile(r"\b[^\W_]+(?:['’\-][^\W_]+)*\b", re.UNICODE)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


def count_words(text: str) -> int:
    """Count Unicode word groups; internal apostrophes/hyphens stay in one word."""

    return len(WORD_PATTERN.findall(text))


def count_bullets(text: str) -> int:
    """Count non-empty lines starting with -, *, •, or a numbered marker."""

    return sum(bool(BULLET_PATTERN.match(line)) for line in text.splitlines())


def check_constraints(
    text: str, constraints: OutputConstraints | None
) -> tuple[ConstraintCheck, ...]:
    """Check deterministic constraints; tone/example requests remain not evaluated."""

    if constraints is None or constraints.is_empty():
        return ()
    checks: list[ConstraintCheck] = []
    folded = text.casefold()
    if constraints.max_words is not None:
        actual = count_words(text)
        checks.append(
            ConstraintCheck(
                rule=f"Maximum {constraints.max_words} words",
                status=CheckStatus.PASS if actual <= constraints.max_words else CheckStatus.FAIL,
                details=f"Counted {actual} words.",
            )
        )
    if constraints.bullet_count is not None:
        actual = count_bullets(text)
        checks.append(
            ConstraintCheck(
                rule=f"Exactly {constraints.bullet_count} bullet points",
                status=CheckStatus.PASS if actual == constraints.bullet_count else CheckStatus.FAIL,
                details=f"Counted {actual} bullet lines.",
            )
        )
    for keyword in constraints.required_keywords:
        present = keyword.casefold() in folded
        checks.append(
            ConstraintCheck(
                rule=f'Include "{keyword}"',
                status=CheckStatus.PASS if present else CheckStatus.FAIL,
                details="Keyword found." if present else "Keyword not found.",
            )
        )
    for phrase in constraints.forbidden_phrases:
        absent = phrase.casefold() not in folded
        checks.append(
            ConstraintCheck(
                rule=f'Exclude "{phrase}"',
                status=CheckStatus.PASS if absent else CheckStatus.FAIL,
                details="Forbidden phrase absent." if absent else "Forbidden phrase found.",
            )
        )
    if constraints.desired_tone:
        checks.append(
            ConstraintCheck(
                rule=f"Tone: {constraints.desired_tone}",
                status=CheckStatus.NOT_EVALUATED,
                details="Tone needs semantic or human evaluation.",
            )
        )
    if constraints.include_example:
        checks.append(
            ConstraintCheck(
                rule="Include an example",
                status=CheckStatus.NOT_EVALUATED,
                details="Example quality/presence is not evaluated deterministically in M3.",
            )
        )
    return tuple(checks)


class TechniqueComparison(BaseModel):
    """Two immutable prompt snapshots suitable for a fair manual comparison."""

    model_config = ConfigDict(frozen=True)

    variant_a: ConstructedTechniquePrompt
    variant_b: ConstructedTechniquePrompt

    @model_validator(mode="after")
    def ensure_common_conditions(self) -> "TechniqueComparison":
        if self.variant_a.task != self.variant_b.task:
            raise ValueError("comparison variants must use the same task")
        if self.variant_a.model_id != self.variant_b.model_id:
            raise ValueError("comparison variants must use the same model")
        return self


def build_comparison(
    builder: TechniquePromptBuilder,
    modified: ConstructedTechniquePrompt,
) -> TechniqueComparison:
    baseline = builder.build(PromptTechnique.ZERO_SHOT, modified.task)
    return TechniqueComparison(variant_a=baseline, variant_b=modified)
