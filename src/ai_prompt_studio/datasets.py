"""Validated local evaluation datasets and safe JSON import."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import PromptValidationError

MAX_DATASET_BYTES = 1_000_000
MAX_TEST_CASES = 100


class DeterministicCheckSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    check: Literal[
        "exact_label",
        "valid_json",
        "pydantic_schema",
        "required_fields",
        "allowed_enum",
        "max_words",
        "required_phrase",
        "forbidden_phrase",
        "nonempty",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    source_text: str | None = None
    input_variables: dict[str, str] = Field(default_factory=dict)
    expected_label: str | None = None
    expected_answer: str | None = None
    deterministic_checks: tuple[DeterministicCheckSpec, ...] = ()
    human_guidance: str | None = None

    @model_validator(mode="after")
    def require_input(self) -> "EvaluationCase":
        if not (self.source_text and self.source_text.strip()) and not self.input_variables:
            raise ValueError("a test case requires source_text or input_variables")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    test_cases: tuple[EvaluationCase, ...] = Field(min_length=1, max_length=MAX_TEST_CASES)
    evaluation_criteria: tuple[str, ...] = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def unique_cases(self) -> "EvaluationDataset":
        ids = [case.case_id for case in self.test_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("test case IDs must be unique")
        return self


def import_dataset_json(payload: str | bytes) -> EvaluationDataset:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > MAX_DATASET_BYTES:
        raise PromptValidationError("Dataset exceeds the 1 MB import limit.")
    try:
        data = json.loads(raw)
        return EvaluationDataset.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
        raise PromptValidationError(
            "Dataset JSON is invalid or does not match the dataset schema."
        ) from None


def load_dataset_file(path: Path) -> EvaluationDataset:
    return import_dataset_json(path.read_bytes())


SEED_DATASET_PATH = Path(__file__).with_name("data") / "sentiment_eval_v1.json"


def load_seed_sentiment_dataset() -> EvaluationDataset:
    return load_dataset_file(SEED_DATASET_PATH)
