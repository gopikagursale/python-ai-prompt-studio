from __future__ import annotations

import sqlite3

import pytest

from ai_prompt_studio.datasets import load_seed_sentiment_dataset
from ai_prompt_studio.evaluation import HumanRating
from ai_prompt_studio.exceptions import PromptValidationError
from ai_prompt_studio.experiments import ExperimentRun, create_experiment_plan
from ai_prompt_studio.persistence import SQLiteRepository
from ai_prompt_studio.versioning import (
    InputVariableDefinition,
    PromptVersionManager,
    VersionStatus,
    utc_now,
)


@pytest.fixture
def repository(tmp_path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "test.db")


@pytest.fixture
def manager(repository) -> PromptVersionManager:
    return PromptVersionManager(repository)


def _initial(manager: PromptVersionManager):
    prompt = manager.create_prompt("Sentiment classifier", "Classify sentiment")
    version = manager.create_initial_version(
        prompt,
        prompt_template="Classify as positive, negative, neutral, or mixed:\n{source_text}",
        developer_instructions="Return only the reviewed sentiment label.",
        input_variables=(InputVariableDefinition(name="source_text", required=True),),
    )
    return prompt, version


def test_prompt_creation_initial_version_and_hash(manager, repository) -> None:
    prompt, version = _initial(manager)

    assert version.version_number == 1
    assert version.status is VersionStatus.PUBLISHED
    assert len(version.content_hash) == 64
    assert repository.get_prompt(prompt.prompt_id).preferred_version_id == version.version_id


def test_draft_editing_and_published_immutability(manager) -> None:
    _, published = _initial(manager)
    draft = manager.create_draft_from_version(published.version_id)
    edited = manager.edit_draft(
        draft.version_id,
        prompt_template="Label the sentiment exactly:\n{source_text}",
        change_description="Make the output instruction more direct",
    )

    assert edited.status is VersionStatus.DRAFT
    assert edited.content_hash != draft.content_hash
    assert manager.require_version(draft.version_id).prompt_template.startswith("Label")
    with pytest.raises(PromptValidationError, match="immutable"):
        manager.edit_draft(published.version_id, prompt_template="overwrite")


def test_publish_new_version_and_restore_older_as_draft(manager) -> None:
    _, first = _initial(manager)
    draft = manager.create_draft_from_version(first.version_id)
    edited = manager.edit_draft(
        draft.version_id, change_description="Candidate", prompt_template="Candidate: {source_text}"
    )
    second = manager.publish_draft(edited.version_id)
    restored = manager.create_draft_from_version(first.version_id)

    assert second.status is VersionStatus.PUBLISHED
    assert second.version_number == 2
    assert manager.repository.get_version(edited.version_id) is None
    assert first.prompt_template != second.prompt_template
    assert restored.version_number == 3
    assert restored.prompt_template == first.prompt_template


def test_compare_and_preferred_version(manager, repository) -> None:
    prompt, first = _initial(manager)
    second = manager.publish_draft(
        manager.edit_draft(
            manager.create_draft_from_version(first.version_id).version_id,
            prompt_template="Candidate: {source_text}",
            change_description="Candidate",
        ).version_id
    )

    differences = manager.compare_versions(first.version_id, second.version_id)
    manager.mark_preferred(second.version_id)

    assert "prompt_template" in differences
    assert repository.get_prompt(prompt.prompt_id).preferred_version_id == second.version_id


def test_database_trigger_blocks_published_updates(manager, repository) -> None:
    _, version = _initial(manager)
    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE prompt_versions SET content_hash='changed' WHERE version_id=?",
                (version.version_id,),
            )


def test_dataset_and_run_persistence_is_idempotent(manager, repository) -> None:
    _, first = _initial(manager)
    second = manager.publish_draft(manager.create_draft_from_version(first.version_id).version_id)
    dataset = load_seed_sentiment_dataset()
    repository.save_dataset(dataset)
    plan = create_experiment_plan("Test", first, second, dataset, "test-model")
    run = ExperimentRun(
        run_id="run-fixed",
        plan=plan,
        started_at=utc_now(),
        completed_at=utc_now(),
        status="completed",
        planned_requests=24,
        actual_api_attempts=0,
        results=(),
    )

    assert repository.save_run(run) is True
    assert repository.save_run(run) is False
    assert repository.get_run("run-fixed") == run
    assert repository.get_dataset(dataset.dataset_id, dataset.version) == dataset
    assert len(repository.list_runs()) == 1


def test_rating_save_and_history_open_do_not_use_api(manager, repository) -> None:
    _, first = _initial(manager)
    second = manager.publish_draft(manager.create_draft_from_version(first.version_id).version_id)
    dataset = load_seed_sentiment_dataset()
    plan = create_experiment_plan("Test", first, second, dataset, "test-model")
    from ai_prompt_studio.experiments import EffectiveRequest, ExperimentCaseResult

    request = EffectiveRequest(
        variant="A",
        prompt_version_id=first.version_id,
        model_id="test-model",
        developer_instructions="trusted",
        user_content="input",
    )
    result = ExperimentCaseResult(
        result_id="result-1",
        run_id="run-1",
        variant="A",
        prompt_version_id=first.version_id,
        case_id="case-1",
        trial=1,
        status="completed",
        api_attempted=True,
        effective_request=request,
        response_text="positive",
    )
    run = ExperimentRun(
        run_id="run-1",
        plan=plan,
        started_at=utc_now(),
        completed_at=utc_now(),
        status="completed",
        planned_requests=1,
        actual_api_attempts=1,
        results=(result,),
    )
    repository.save_run(run)
    rating = HumanRating(result_id="result-1", accuracy=5, notes="Reviewed")

    repository.save_rating(rating)
    loaded = repository.get_run("run-1")

    assert loaded.results[0].response_text == "positive"
    assert repository.get_ratings("run-1")["result-1"] == rating

    repository.delete_run("run-1")

    assert repository.get_run("run-1") is None
    assert repository.get_ratings("run-1") == {}


def test_parameterized_prompt_storage_treats_sql_as_data(manager, repository) -> None:
    malicious_name = "name'); DROP TABLE prompts; --"
    prompt = manager.create_prompt(malicious_name, "SQL-looking text is data")

    assert repository.get_prompt(prompt.prompt_id).name == malicious_name
    assert len(repository.list_prompts()) == 1
