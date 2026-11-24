"""Sequential, explicitly confirmed local A/B experiment runner."""

from __future__ import annotations

from datetime import datetime
from string import Formatter
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .client import OpenAIModelClient
from .datasets import EvaluationCase, EvaluationDataset
from .evaluation import EvaluationRecord, evaluate_case_output
from .exceptions import PromptStudioError, PromptValidationError
from .models import UsageMetadata
from .structured_outputs import (
    StructuredOutputClient,
    StructuredPromptPreview,
    get_output_schema,
)
from .versioning import PromptVersion, VersionStatus, utc_now

MAX_EXPERIMENT_REQUESTS = 100


class GenerationSettings(BaseModel):
    """Common request settings; M5 intentionally uses SDK defaults for generation."""

    model_config = ConfigDict(frozen=True)

    store: Literal[False] = False


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    name: str = Field(min_length=1)
    baseline: PromptVersion
    candidate: PromptVersion
    dataset: EvaluationDataset
    model_id: str = Field(min_length=1)
    generation_settings: GenerationSettings = GenerationSettings()
    trial_count: int = Field(default=1, ge=1, le=3)
    request_cap: int = Field(default=MAX_EXPERIMENT_REQUESTS, ge=1, le=MAX_EXPERIMENT_REQUESTS)
    created_at: datetime

    @model_validator(mode="after")
    def validate_plan(self) -> "ExperimentPlan":
        if (
            self.baseline.status is not VersionStatus.PUBLISHED
            or self.candidate.status is not VersionStatus.PUBLISHED
        ):
            raise ValueError("experiments require published prompt versions")
        if self.baseline.version_id == self.candidate.version_id:
            raise ValueError("baseline and candidate must be different versions")
        if self.planned_requests > self.request_cap:
            raise ValueError("planned request count exceeds the configured cap")
        return self

    @property
    def planned_requests(self) -> int:
        return len(self.dataset.test_cases) * 2 * self.trial_count

    @property
    def schema_compatible(self) -> bool:
        return (
            self.baseline.structured_output_schema_id == self.candidate.structured_output_schema_id
        )


def create_experiment_plan(
    name: str,
    baseline: PromptVersion,
    candidate: PromptVersion,
    dataset: EvaluationDataset,
    model_id: str,
    *,
    trial_count: int = 1,
    request_cap: int = MAX_EXPERIMENT_REQUESTS,
) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id=f"experiment_{uuid4().hex}",
        name=name,
        baseline=baseline,
        candidate=candidate,
        dataset=dataset,
        model_id=model_id,
        trial_count=trial_count,
        request_cap=request_cap,
        created_at=utc_now(),
    )


class EffectiveRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: Literal["A", "B"]
    prompt_version_id: str
    model_id: str
    developer_instructions: str
    user_content: str
    few_shot_examples: tuple[dict[str, str], ...] = ()
    constraints: tuple[str, ...] = ()
    structured_output_schema_id: str | None = None


class ExperimentCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    result_id: str
    run_id: str
    variant: Literal["A", "B"]
    prompt_version_id: str
    case_id: str
    trial: int
    status: Literal["completed", "failed"]
    api_attempted: bool
    effective_request: EffectiveRequest
    expected_label: str | None = None
    response_text: str | None = None
    structured_data: dict[str, Any] | None = None
    usage: UsageMetadata | None = None
    latency_seconds: float | None = None
    error_message: str | None = None
    evaluations: tuple[EvaluationRecord, ...] = ()


class ExperimentRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    plan: ExperimentPlan
    started_at: datetime
    completed_at: datetime
    status: Literal["completed", "partial", "failed"]
    planned_requests: int
    actual_api_attempts: int
    results: tuple[ExperimentCaseResult, ...]


def render_effective_request(
    version: PromptVersion,
    case: EvaluationCase,
    model_id: str,
    variant: Literal["A", "B"],
) -> EffectiveRequest:
    values = dict(case.input_variables)
    if case.source_text is not None:
        values.setdefault("source_text", case.source_text)
    required_fields: list[str] = []
    try:
        for _, field_name, format_spec, conversion in Formatter().parse(version.prompt_template):
            if field_name is None:
                continue
            if not field_name.isidentifier() or format_spec or conversion:
                raise PromptValidationError("Prompt version contains unsupported template syntax.")
            required_fields.append(field_name)
    except ValueError:
        raise PromptValidationError("Prompt version template is invalid.") from None
    missing = sorted(set(required_fields) - set(values))
    if missing:
        raise PromptValidationError(f"Test case is missing variables: {', '.join(missing)}.")
    user_content = version.prompt_template.format_map(values)
    examples_raw = version.technique_settings.get("examples", [])
    examples: list[dict[str, str]] = []
    if examples_raw:
        blocks: list[str] = []
        for index, item in enumerate(examples_raw, start=1):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("input"), str)
                or not isinstance(item.get("output"), str)
            ):
                raise PromptValidationError("Prompt version has invalid few-shot examples.")
            example = {"input": item["input"], "output": item["output"]}
            examples.append(example)
            blocks.append(
                f"Demonstration {index}\nInput: {example['input']}\nExpected output: {example['output']}"
            )
        user_content = (
            "User-controlled demonstrations:\n\n"
            + "\n\n".join(blocks)
            + "\n\n---\nNew task:\n"
            + user_content
        )
    if version.output_constraints:
        user_content += "\n\nOutput constraints:\n- " + "\n- ".join(version.output_constraints)
    return EffectiveRequest(
        variant=variant,
        prompt_version_id=version.version_id,
        model_id=model_id,
        developer_instructions=version.developer_instructions,
        user_content=user_content,
        few_shot_examples=tuple(examples),
        constraints=version.output_constraints,
        structured_output_schema_id=version.structured_output_schema_id,
    )


class ExperimentRunner:
    def __init__(
        self,
        text_client: OpenAIModelClient,
        structured_client: StructuredOutputClient | None = None,
    ) -> None:
        self.text_client = text_client
        self.structured_client = structured_client

    def run(self, plan: ExperimentPlan, *, confirmed: bool) -> ExperimentRun:
        if not confirmed:
            raise PromptValidationError("Explicit experiment confirmation is required.")
        if self.text_client.model != plan.model_id:
            raise PromptValidationError("Experiment model must match the configured client model.")
        run_id = f"run_{uuid4().hex}"
        started = utc_now()
        results: list[ExperimentCaseResult] = []
        attempts = 0
        for case in plan.dataset.test_cases:
            for trial in range(1, plan.trial_count + 1):
                for variant, version in (("A", plan.baseline), ("B", plan.candidate)):
                    request = render_effective_request(version, case, plan.model_id, variant)
                    attempts += 1
                    results.append(self._execute_one(run_id, request, case, trial))
        completed = sum(result.status == "completed" for result in results)
        status = (
            "completed" if completed == len(results) else "failed" if completed == 0 else "partial"
        )
        return ExperimentRun(
            run_id=run_id,
            plan=plan,
            started_at=started,
            completed_at=utc_now(),
            status=status,
            planned_requests=plan.planned_requests,
            actual_api_attempts=attempts,
            results=tuple(results),
        )

    def _execute_one(
        self,
        run_id: str,
        request: EffectiveRequest,
        case: EvaluationCase,
        trial: int,
    ) -> ExperimentCaseResult:
        result_id = f"result_{uuid4().hex}"
        try:
            if request.structured_output_schema_id:
                if self.structured_client is None:
                    raise PromptValidationError(
                        "A structured-output client is required for this prompt version."
                    )
                definition = get_output_schema(request.structured_output_schema_id)
                preview = StructuredPromptPreview(
                    schema_id=definition.schema_id,
                    schema_name=definition.display_name,
                    model_id=request.model_id,
                    developer_instructions=request.developer_instructions,
                    user_content=request.user_content,
                    json_schema=definition.model.model_json_schema(),
                )
                generated = self.structured_client.generate_structured(preview)
                response_text = generated.raw_json
                structured_data = generated.parsed.model_dump(mode="json")
                usage, latency = generated.usage, generated.latency_seconds
            else:
                from .models import RenderedPrompt

                prompt = RenderedPrompt(
                    template_id=request.prompt_version_id,
                    template_name=f"Experiment variant {request.variant}",
                    template_version="1.0.0",
                    developer_instructions=request.developer_instructions,
                    user_prompt=request.user_content,
                )
                started = perf_counter()
                generated = self.text_client.generate(prompt)
                latency = max(0.0, perf_counter() - started)
                response_text, structured_data, usage = generated.text, None, generated.usage
            evaluations = evaluate_case_output(
                case,
                response_text,
                structured_data=structured_data,
                structured_schema_id=request.structured_output_schema_id,
            )
            return ExperimentCaseResult(
                result_id=result_id,
                run_id=run_id,
                variant=request.variant,
                prompt_version_id=request.prompt_version_id,
                case_id=case.case_id,
                trial=trial,
                status="completed",
                api_attempted=True,
                effective_request=request,
                expected_label=case.expected_label,
                response_text=response_text,
                structured_data=structured_data,
                usage=usage,
                latency_seconds=latency,
                evaluations=evaluations,
            )
        except PromptStudioError as exc:
            message = str(exc)
        except Exception:
            message = "Unexpected experiment request failure."
        return ExperimentCaseResult(
            result_id=result_id,
            run_id=run_id,
            variant=request.variant,
            prompt_version_id=request.prompt_version_id,
            case_id=case.case_id,
            trial=trial,
            status="failed",
            api_attempted=True,
            effective_request=request,
            expected_label=case.expected_label,
            error_message=message,
        )
