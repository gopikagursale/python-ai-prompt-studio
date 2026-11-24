"""Secret-free JSON/CSV export for local experiment results."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .evaluation import HumanRating
from .experiments import ExperimentRun

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def safe_csv_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def export_run_csv(run: ExperimentRun, ratings: dict[str, HumanRating] | None = None) -> str:
    ratings = ratings or {}
    output = io.StringIO(newline="")
    fields = [
        "experiment_id",
        "run_id",
        "timestamp",
        "variant",
        "prompt_version_id",
        "model_id",
        "dataset_id",
        "dataset_version",
        "test_case_id",
        "trial",
        "expected_result",
        "generated_result",
        "status",
        "evaluation_outcomes",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
        "human_accuracy",
        "human_relevance",
        "human_clarity",
        "human_instruction_following",
        "reviewer_notes",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in run.results:
        rating = ratings.get(result.result_id)
        row = {
            "experiment_id": run.plan.experiment_id,
            "run_id": run.run_id,
            "timestamp": run.completed_at.isoformat(),
            "variant": result.variant,
            "prompt_version_id": result.prompt_version_id,
            "model_id": run.plan.model_id,
            "dataset_id": run.plan.dataset.dataset_id,
            "dataset_version": run.plan.dataset.version,
            "test_case_id": result.case_id,
            "trial": result.trial,
            "expected_result": result.expected_label or "",
            "generated_result": result.response_text or result.error_message or "",
            "status": result.status,
            "evaluation_outcomes": "; ".join(
                f"{item.evaluation_name}:{item.outcome.value}" for item in result.evaluations
            ),
            "input_tokens": result.usage.input_tokens if result.usage else "",
            "output_tokens": result.usage.output_tokens if result.usage else "",
            "total_tokens": result.usage.total_tokens if result.usage else "",
            "latency_seconds": result.latency_seconds if result.latency_seconds is not None else "",
            "human_accuracy": rating.accuracy if rating else "",
            "human_relevance": rating.relevance if rating else "",
            "human_clarity": rating.clarity if rating else "",
            "human_instruction_following": rating.instruction_following if rating else "",
            "reviewer_notes": rating.notes if rating else "",
        }
        writer.writerow({key: safe_csv_cell(value) for key, value in row.items()})
    return output.getvalue()


def export_run_json(run: ExperimentRun, ratings: dict[str, HumanRating] | None = None) -> str:
    payload = {
        "experiment": run.model_dump(mode="json"),
        "human_ratings": {
            result_id: rating.model_dump(mode="json")
            for result_id, rating in (ratings or {}).items()
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
