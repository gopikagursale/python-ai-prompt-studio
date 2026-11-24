from __future__ import annotations

import csv
import io
import json
from unittest.mock import Mock

import pytest

from ai_prompt_studio.datasets import EvaluationDataset, load_seed_sentiment_dataset
from ai_prompt_studio.exceptions import ModelClientError, PromptValidationError
from ai_prompt_studio.experiments import (
    ExperimentRunner,
    create_experiment_plan,
    render_effective_request,
)
from ai_prompt_studio.exports import export_run_csv, export_run_json, safe_csv_cell
from ai_prompt_studio.models import GenerationResult, UsageMetadata
from ai_prompt_studio.persistence import SQLiteRepository
from ai_prompt_studio.versioning import InputVariableDefinition, PromptVersionManager


def _versions(tmp_path):
    repo = SQLiteRepository(tmp_path / "x.db")
    manager = PromptVersionManager(repo)
    prompt = manager.create_prompt("Sentiment", "Test")
    a = manager.create_initial_version(
        prompt,
        prompt_template="Review:\n{source_text}\nReturn one label.",
        developer_instructions="Classify sentiment.",
        input_variables=(InputVariableDefinition(name="source_text"),),
    )
    draft = manager.create_draft_from_version(a.version_id)
    draft = manager.edit_draft(
        draft.version_id,
        prompt_template="Classify this review as positive, negative, neutral, or mixed:\n{source_text}",
        change_description="Explicit labels",
    )
    b = manager.publish_draft(draft.version_id)
    return a, b


def _small_dataset() -> EvaluationDataset:
    seed = load_seed_sentiment_dataset()
    return seed.model_copy(update={"test_cases": seed.test_cases[:2]})


def test_request_count_and_plan_controls(tmp_path) -> None:
    a, b = _versions(tmp_path)
    plan = create_experiment_plan("AB", a, b, _small_dataset(), "model", trial_count=2)
    assert plan.planned_requests == 8
    assert plan.schema_compatible
    with pytest.raises(ValueError, match="cap"):
        create_experiment_plan(
            "too large", a, b, load_seed_sentiment_dataset(), "model", trial_count=3, request_cap=10
        )


def test_effective_prompt_preserves_roles_examples_and_constraints(tmp_path) -> None:
    a, _ = _versions(tmp_path)
    data = a.model_dump()
    data["technique_settings"] = {"examples": [{"input": "great", "output": "positive"}]}
    data["output_constraints"] = ("Return exactly one label.",)
    from ai_prompt_studio.versioning import PromptVersion, prompt_content_hash

    data["content_hash"] = prompt_content_hash(data)
    version = PromptVersion.model_validate(data)
    case = _small_dataset().test_cases[0]

    request = render_effective_request(version, case, "model", "A")

    assert case.source_text in request.user_content
    assert case.source_text not in request.developer_instructions
    assert "Demonstration 1" in request.user_content
    assert request.constraints == ("Return exactly one label.",)


def test_execution_requires_confirmation_and_uses_same_model_order(tmp_path) -> None:
    a, b = _versions(tmp_path)
    dataset = _small_dataset()
    plan = create_experiment_plan("AB", a, b, dataset, "model")
    client = Mock(model="model")
    client.generate.side_effect = [
        GenerationResult(text="positive"),
        GenerationResult(text="positive"),
        GenerationResult(text="positive"),
        GenerationResult(text="positive"),
    ]
    runner = ExperimentRunner(client)

    with pytest.raises(PromptValidationError, match="confirmation"):
        runner.run(plan, confirmed=False)
    run = runner.run(plan, confirmed=True)

    assert run.actual_api_attempts == 4
    assert [result.variant for result in run.results] == ["A", "B", "A", "B"]
    assert [result.case_id for result in run.results] == [dataset.test_cases[0].case_id] * 2 + [
        dataset.test_cases[1].case_id
    ] * 2
    assert all(result.effective_request.model_id == "model" for result in run.results)


def test_partial_failures_preserve_successful_results(tmp_path) -> None:
    a, b = _versions(tmp_path)
    one_case = _small_dataset().model_copy(update={"test_cases": _small_dataset().test_cases[:1]})
    plan = create_experiment_plan("AB", a, b, one_case, "model")
    client = Mock(model="model")
    client.generate.side_effect = [
        GenerationResult(text="positive"),
        ModelClientError("Safe failure"),
    ]

    run = ExperimentRunner(client).run(plan, confirmed=True)

    assert run.status == "partial"
    assert run.results[0].status == "completed"
    assert run.results[1].status == "failed"
    assert run.results[1].error_message == "Safe failure"


def test_csv_formula_safety_and_json_integrity(tmp_path) -> None:
    a, b = _versions(tmp_path)
    dataset = _small_dataset().model_copy(update={"test_cases": _small_dataset().test_cases[:1]})
    plan = create_experiment_plan("AB", a, b, dataset, "model")
    client = Mock(model="model")
    client.generate.side_effect = [
        GenerationResult(
            text='=HYPERLINK("bad")',
            usage=UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
        GenerationResult(text="positive"),
    ]
    run = ExperimentRunner(client).run(plan, confirmed=True)

    csv_text = export_run_csv(run)
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert rows[0]["generated_result"].startswith("'=HYPERLINK")
    payload = json.loads(export_run_json(run))
    assert payload["experiment"]["run_id"] == run.run_id
    assert len(payload["experiment"]["results"]) == 2
    assert safe_csv_cell("+cmd") == "'+cmd"
