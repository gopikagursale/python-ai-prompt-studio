from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from ai_prompt_studio.client import OpenAIModelClient
from ai_prompt_studio.config import AppSettings
from ai_prompt_studio.datasets import (
    MAX_DATASET_BYTES,
    import_dataset_json,
    load_seed_sentiment_dataset,
)
from ai_prompt_studio.exceptions import ModelClientError, PromptValidationError
from ai_prompt_studio.experiments import ExperimentRunner, create_experiment_plan
from ai_prompt_studio.exports import export_run_csv, export_run_json
from ai_prompt_studio.models import GenerationResult, RenderedPrompt
from ai_prompt_studio.persistence import SQLiteRepository
from ai_prompt_studio.versioning import InputVariableDefinition, PromptVersionManager


def test_raw_provider_exception_secret_is_not_exposed() -> None:
    secret = "credential-that-must-not-appear"
    sdk = Mock()
    sdk.responses.create.side_effect = RuntimeError(secret)
    settings = AppSettings(api_key=secret, model="test-model")
    prompt = RenderedPrompt(
        template_id="security",
        template_name="Security",
        template_version="1.0.0",
        developer_instructions="Trusted instructions",
        user_prompt="Untrusted input",
    )

    with pytest.raises(ModelClientError) as caught:
        OpenAIModelClient(settings, sdk_client=sdk).generate(prompt)

    assert secret not in str(caught.value)


def test_imported_dataset_rejects_oversized_and_unknown_content() -> None:
    with pytest.raises(PromptValidationError, match="1 MB"):
        import_dataset_json(b"x" * (MAX_DATASET_BYTES + 1))

    payload = json.dumps(
        {
            "dataset_id": "unsafe",
            "name": "Unsafe",
            "description": "Unknown fields must be rejected",
            "version": "1.0.0",
            "test_cases": [
                {
                    "case_id": "case-1",
                    "source_text": "__import__('os').system('echo unsafe')",
                    "unexpected_code": "never execute me",
                }
            ],
            "evaluation_criteria": ["nonempty"],
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    with pytest.raises(PromptValidationError):
        import_dataset_json(payload)


def test_exports_contain_results_but_not_application_credentials(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "security.db")
    manager = PromptVersionManager(repository)
    prompt = manager.create_prompt("Sentiment")
    baseline = manager.create_initial_version(
        prompt,
        prompt_template="Classify: {source_text}",
        developer_instructions="Return one sentiment label.",
        input_variables=(InputVariableDefinition(name="source_text"),),
    )
    candidate = manager.publish_draft(
        manager.create_draft_from_version(baseline.version_id).version_id
    )

    dataset = load_seed_sentiment_dataset().model_copy(
        update={"test_cases": load_seed_sentiment_dataset().test_cases[:1]}
    )
    plan = create_experiment_plan("Security export", baseline, candidate, dataset, "test-model")
    client = Mock(model="test-model")
    client.generate.side_effect = [
        GenerationResult(text="positive"),
        GenerationResult(text="positive"),
    ]
    run = ExperimentRunner(client).run(plan, confirmed=True)

    exported = export_run_csv(run) + export_run_json(run)

    assert "positive" in exported
    assert "OPENAI_API_KEY" not in exported
    assert "credential-that-must-not-appear" not in exported
