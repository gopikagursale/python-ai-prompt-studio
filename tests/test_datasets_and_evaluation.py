from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_prompt_studio.datasets import (
    DeterministicCheckSpec,
    EvaluationCase,
    import_dataset_json,
    load_seed_sentiment_dataset,
)
from ai_prompt_studio.evaluation import (
    EvaluationOutcome,
    HumanRating,
    calculate_variant_metrics,
    detect_regressions,
    evaluate_case_output,
    parse_sentiment_label,
)
from ai_prompt_studio.exceptions import PromptValidationError
from ai_prompt_studio.experiments import EffectiveRequest, ExperimentCaseResult
from ai_prompt_studio.models import UsageMetadata


def test_seed_dataset_has_reviewed_twelve_cases() -> None:
    dataset = load_seed_sentiment_dataset()

    assert len(dataset.test_cases) == 12
    assert {case.expected_label for case in dataset.test_cases} == {
        "positive",
        "negative",
        "neutral",
        "mixed",
    }
    assert len({case.case_id for case in dataset.test_cases}) == 12


def test_dataset_import_validation_and_size_limit() -> None:
    dataset = load_seed_sentiment_dataset()
    imported = import_dataset_json(dataset.model_dump_json())
    assert imported == dataset
    with pytest.raises(PromptValidationError):
        import_dataset_json("not json")
    with pytest.raises(PromptValidationError, match="1 MB"):
        import_dataset_json(b"x" * 1_000_001)


def test_missing_expected_label_is_not_applicable() -> None:
    case = EvaluationCase(
        case_id="open",
        source_text="Text",
        deterministic_checks=(DeterministicCheckSpec(check="exact_label"),),
    )
    record = evaluate_case_output(case, "positive")[0]
    assert record.outcome is EvaluationOutcome.NOT_APPLICABLE
    assert record.applicable is False


def test_exact_match_and_unparseable_classification() -> None:
    case = EvaluationCase(
        case_id="one",
        source_text="Great",
        expected_label="positive",
        deterministic_checks=(DeterministicCheckSpec(check="exact_label"),),
    )
    assert evaluate_case_output(case, "Positive")[0].outcome is EvaluationOutcome.PASS
    assert (
        evaluate_case_output(case, "The sentiment is positive because it is good.")[0].outcome
        is EvaluationOutcome.ERROR
    )
    assert parse_sentiment_label("negative.") == "negative"
    assert parse_sentiment_label("positive and negative") is None


def test_json_pydantic_and_constraint_checks() -> None:
    case = EvaluationCase(
        case_id="checks",
        source_text="Text",
        deterministic_checks=(
            DeterministicCheckSpec(check="valid_json"),
            DeterministicCheckSpec(check="required_fields", parameters={"fields": ["sentiment"]}),
            DeterministicCheckSpec(
                check="allowed_enum",
                parameters={"field": "sentiment", "allowed": ["positive", "negative"]},
            ),
            DeterministicCheckSpec(check="max_words", parameters={"value": 3}),
            DeterministicCheckSpec(check="required_phrase", parameters={"value": "positive"}),
            DeterministicCheckSpec(check="forbidden_phrase", parameters={"value": "secret"}),
        ),
    )
    response = '{"sentiment":"positive"}'
    records = evaluate_case_output(case, response, structured_data={"sentiment": "positive"})
    outcomes = {record.evaluation_name: record.outcome for record in records}
    assert outcomes["valid_json"] is EvaluationOutcome.PASS
    assert outcomes["required_fields"] is EvaluationOutcome.PASS
    assert outcomes["allowed_enum"] is EvaluationOutcome.PASS
    assert outcomes["required_phrase"] is EvaluationOutcome.PASS
    assert outcomes["forbidden_phrase"] is EvaluationOutcome.PASS


@pytest.mark.parametrize("field", ["accuracy", "relevance", "clarity", "instruction_following"])
def test_human_rating_validation(field: str) -> None:
    with pytest.raises(ValidationError):
        HumanRating(result_id="r", **{field: 6})
    assert getattr(HumanRating(result_id="r", **{field: 1}), field) == 1


def _result(result_id: str, variant: str, case_id: str, outcome, *, status="completed", usage=True):
    request = EffectiveRequest(
        variant=variant,
        prompt_version_id=f"v-{variant}",
        model_id="m",
        developer_instructions="d",
        user_content="u",
    )
    case = EvaluationCase(
        case_id=case_id,
        source_text="x",
        expected_label="positive",
        deterministic_checks=(DeterministicCheckSpec(check="exact_label"),),
    )
    evaluations = (
        evaluate_case_output(case, "positive" if outcome else "negative")
        if status == "completed"
        else ()
    )
    return ExperimentCaseResult(
        result_id=result_id,
        run_id="run",
        variant=variant,
        prompt_version_id=f"v-{variant}",
        case_id=case_id,
        trial=1,
        status=status,
        api_attempted=True,
        effective_request=request,
        response_text="positive" if outcome else "negative" if status == "completed" else None,
        usage=UsageMetadata(input_tokens=2, output_tokens=1, total_tokens=3)
        if usage and status == "completed"
        else None,
        latency_seconds=0.2 if status == "completed" else None,
        evaluations=evaluations,
    )


def test_metrics_use_planned_denominator_and_preserve_missing_values() -> None:
    results = [
        _result("a1", "A", "c1", True),
        _result("a2", "A", "c2", False, status="failed", usage=False),
    ]
    metrics = calculate_variant_metrics(results, "A", planned_count=2)
    assert metrics.completed_count == 1
    assert metrics.failed_count == 1
    assert metrics.exact_match_correct == 1
    assert metrics.exact_match_accuracy == 0.5
    assert metrics.completed_only_accuracy == 1.0
    assert metrics.total_tokens == 3


def test_regression_detection() -> None:
    results = [
        _result("a1", "A", "both", True),
        _result("b1", "B", "both", True),
        _result("a2", "A", "regress", True),
        _result("b2", "B", "regress", False),
        _result("a3", "A", "improve", False),
        _result("b3", "B", "improve", True),
        _result("a4", "A", "neither", False),
        _result("b4", "B", "neither", False),
    ]
    summary = detect_regressions(results)
    assert summary.candidate_regressions == ("regress / trial 1",)
    assert summary.b_correct_a_incorrect == ("improve / trial 1",)
