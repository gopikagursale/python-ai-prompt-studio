"""Deterministic and human evaluation for local prompt experiments."""

from __future__ import annotations

import json
import re
from enum import Enum
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datasets import EvaluationCase
from .structured_outputs import get_output_schema
from .techniques import count_words

SENTIMENT_LABELS = {"positive", "negative", "neutral", "mixed"}


class EvaluationOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluation_name: str
    applicable: bool
    outcome: EvaluationOutcome
    expected_value: Any = None
    actual_value: Any = None
    details: str


def parse_sentiment_label(
    response_text: str, structured_data: dict[str, Any] | None = None
) -> str | None:
    if structured_data is not None:
        value = structured_data.get("sentiment")
        return (
            value.casefold()
            if isinstance(value, str) and value.casefold() in SENTIMENT_LABELS
            else None
        )
    normalized = response_text.strip().casefold()
    match = re.fullmatch(r"(?:sentiment\s*:\s*)?(positive|negative|neutral|mixed)[.!]?", normalized)
    return match.group(1) if match else None


def evaluate_case_output(
    case: EvaluationCase,
    response_text: str,
    *,
    structured_data: dict[str, Any] | None = None,
    structured_schema_id: str | None = None,
) -> tuple[EvaluationRecord, ...]:
    records: list[EvaluationRecord] = []
    for spec in case.deterministic_checks:
        name, params = spec.check, spec.parameters
        if name == "exact_label":
            if case.expected_label is None:
                records.append(
                    EvaluationRecord(
                        evaluation_name=name,
                        applicable=False,
                        outcome=EvaluationOutcome.NOT_APPLICABLE,
                        details="No reviewed expected label was provided.",
                    )
                )
                continue
            actual = parse_sentiment_label(response_text, structured_data)
            records.append(
                EvaluationRecord(
                    evaluation_name=name,
                    applicable=True,
                    outcome=(
                        EvaluationOutcome.ERROR
                        if actual is None
                        else EvaluationOutcome.PASS
                        if actual == case.expected_label.casefold()
                        else EvaluationOutcome.FAIL
                    ),
                    expected_value=case.expected_label.casefold(),
                    actual_value=actual,
                    details="Classification was unparseable."
                    if actual is None
                    else "Exact reviewed-label comparison.",
                )
            )
        elif name == "nonempty":
            valid = bool(response_text.strip()) or structured_data is not None
            records.append(_boolean_record(name, valid, True, "Response must contain output."))
        elif name == "valid_json":
            try:
                json.loads(response_text)
                valid = True
            except (json.JSONDecodeError, TypeError):
                valid = False
            records.append(_boolean_record(name, valid, True, "JSON syntax validation."))
        elif name == "pydantic_schema":
            if not structured_schema_id:
                records.append(
                    EvaluationRecord(
                        evaluation_name=name,
                        applicable=False,
                        outcome=EvaluationOutcome.NOT_APPLICABLE,
                        details="No structured schema is configured.",
                    )
                )
            else:
                try:
                    data = (
                        structured_data
                        if structured_data is not None
                        else json.loads(response_text)
                    )
                    get_output_schema(structured_schema_id).model.model_validate(data)
                    valid, details = True, "Pydantic schema validation succeeded."
                except Exception:
                    valid, details = False, "Pydantic schema validation failed."
                records.append(_boolean_record(name, valid, True, details))
        elif name == "required_fields":
            fields = tuple(params.get("fields", ()))
            data = structured_data
            if data is None:
                try:
                    data = json.loads(response_text)
                except (json.JSONDecodeError, TypeError):
                    data = None
            missing = [field for field in fields if not isinstance(data, dict) or field not in data]
            records.append(
                _boolean_record(
                    name,
                    not missing,
                    fields,
                    f"Missing fields: {', '.join(missing)}"
                    if missing
                    else "All required fields are present.",
                )
            )
        elif name == "allowed_enum":
            field = str(params.get("field", ""))
            allowed = tuple(params.get("allowed", ()))
            data = structured_data or {}
            actual = data.get(field) if isinstance(data, dict) else None
            records.append(
                EvaluationRecord(
                    evaluation_name=name,
                    applicable=bool(field and allowed),
                    outcome=(
                        EvaluationOutcome.PASS if actual in allowed else EvaluationOutcome.FAIL
                    ),
                    expected_value=allowed,
                    actual_value=actual,
                    details=f"Allowed values for {field}.",
                )
            )
        elif name == "max_words":
            limit = int(params.get("value", 0))
            actual = count_words(response_text)
            records.append(
                EvaluationRecord(
                    evaluation_name=name,
                    applicable=limit > 0,
                    outcome=EvaluationOutcome.PASS
                    if limit > 0 and actual <= limit
                    else EvaluationOutcome.FAIL,
                    expected_value=limit,
                    actual_value=actual,
                    details="Unicode word-count comparison.",
                )
            )
        elif name in {"required_phrase", "forbidden_phrase"}:
            phrase = str(params.get("value", ""))
            present = phrase.casefold() in response_text.casefold()
            passed = present if name == "required_phrase" else not present
            records.append(_boolean_record(name, passed, phrase, "Case-insensitive phrase check."))
    return tuple(records)


def _boolean_record(name: str, passed: bool, expected: Any, details: str) -> EvaluationRecord:
    return EvaluationRecord(
        evaluation_name=name,
        applicable=True,
        outcome=EvaluationOutcome.PASS if passed else EvaluationOutcome.FAIL,
        expected_value=expected,
        actual_value=passed,
        details=details,
    )


RATING_DEFINITIONS = {
    1: "Incorrect or unusable",
    2: "Major problems",
    3: "Acceptable with notable issues",
    4: "Good with minor issues",
    5: "Excellent",
}


class HumanRating(BaseModel):
    model_config = ConfigDict(frozen=True)

    result_id: str
    accuracy: int | None = Field(default=None, ge=1, le=5)
    relevance: int | None = Field(default=None, ge=1, le=5)
    clarity: int | None = Field(default=None, ge=1, le=5)
    instruction_following: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""
    reviewer_blinded: bool = False

    @model_validator(mode="after")
    def require_score_or_notes(self) -> "HumanRating":
        scores = (self.accuracy, self.relevance, self.clarity, self.instruction_following)
        if all(score is None for score in scores) and not self.notes.strip():
            raise ValueError("a rating requires at least one score or reviewer notes")
        return self


class VariantMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str
    planned_count: int
    completed_count: int
    failed_count: int
    exact_match_correct: int | None
    exact_match_denominator: int | None
    exact_match_accuracy: float | None
    completed_only_accuracy: float | None
    format_pass_rate: float | None
    constraint_pass_rate: float | None
    average_latency_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    human_score_averages: dict[str, float] = Field(default_factory=dict)


def calculate_variant_metrics(
    results: list[Any],
    variant: str,
    planned_count: int,
    ratings: dict[str, HumanRating] | None = None,
) -> VariantMetrics:
    selected = [result for result in results if result.variant == variant]
    completed = [result for result in selected if result.status == "completed"]
    failed = len(selected) - len(completed)
    exact = [
        evaluation
        for result in selected
        for evaluation in result.evaluations
        if evaluation.evaluation_name == "exact_label" and evaluation.applicable
    ]
    correct = sum(evaluation.outcome is EvaluationOutcome.PASS for evaluation in exact)
    parseable = [
        evaluation for evaluation in exact if evaluation.outcome is not EvaluationOutcome.ERROR
    ]
    format_evals = [
        evaluation
        for result in completed
        for evaluation in result.evaluations
        if evaluation.evaluation_name
        in {"valid_json", "pydantic_schema", "required_fields", "allowed_enum"}
        and evaluation.applicable
    ]
    constraint_evals = [
        evaluation
        for result in completed
        for evaluation in result.evaluations
        if evaluation.evaluation_name in {"max_words", "required_phrase", "forbidden_phrase"}
        and evaluation.applicable
    ]
    latencies = [
        result.latency_seconds for result in completed if result.latency_seconds is not None
    ]
    usage_items = [result.usage for result in completed if result.usage is not None]
    human: dict[str, float] = {}
    if ratings:
        relevant = [ratings[result.result_id] for result in selected if result.result_id in ratings]
        for field in ("accuracy", "relevance", "clarity", "instruction_following"):
            values = [
                getattr(rating, field) for rating in relevant if getattr(rating, field) is not None
            ]
            if values:
                human[field] = mean(values)
    return VariantMetrics(
        variant=variant,
        planned_count=planned_count,
        completed_count=len(completed),
        failed_count=failed,
        exact_match_correct=correct if exact else None,
        exact_match_denominator=planned_count if exact else None,
        exact_match_accuracy=(correct / planned_count if exact and planned_count else None),
        completed_only_accuracy=(correct / len(parseable) if parseable else None),
        format_pass_rate=_pass_rate(format_evals),
        constraint_pass_rate=_pass_rate(constraint_evals),
        average_latency_seconds=mean(latencies) if latencies else None,
        input_tokens=sum(item.input_tokens for item in usage_items) if usage_items else None,
        output_tokens=sum(item.output_tokens for item in usage_items) if usage_items else None,
        total_tokens=sum(item.total_tokens for item in usage_items) if usage_items else None,
        human_score_averages=human,
    )


def _pass_rate(records: list[EvaluationRecord]) -> float | None:
    return (
        (sum(record.outcome is EvaluationOutcome.PASS for record in records) / len(records))
        if records
        else None
    )


class RegressionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    both_correct: tuple[str, ...]
    a_correct_b_incorrect: tuple[str, ...]
    b_correct_a_incorrect: tuple[str, ...]
    both_incorrect: tuple[str, ...]
    candidate_regressions: tuple[str, ...]


def detect_regressions(results: list[Any]) -> RegressionSummary:
    paired: dict[tuple[str, int], dict[str, bool]] = {}
    for result in results:
        exact = next(
            (item for item in result.evaluations if item.evaluation_name == "exact_label"), None
        )
        paired.setdefault((result.case_id, result.trial), {})[result.variant] = bool(
            exact and exact.outcome is EvaluationOutcome.PASS
        )
    groups = {"both": [], "a_only": [], "b_only": [], "neither": []}
    for (case_id, trial), outcomes in sorted(paired.items()):
        label = f"{case_id} / trial {trial}"
        a, b = outcomes.get("A", False), outcomes.get("B", False)
        groups["both" if a and b else "a_only" if a else "b_only" if b else "neither"].append(label)
    return RegressionSummary(
        both_correct=tuple(groups["both"]),
        a_correct_b_incorrect=tuple(groups["a_only"]),
        b_correct_a_incorrect=tuple(groups["b_only"]),
        both_incorrect=tuple(groups["neither"]),
        candidate_regressions=tuple(groups["a_only"]),
    )
