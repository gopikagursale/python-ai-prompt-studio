"""Local single-user SQLite persistence with parameterized queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .datasets import EvaluationDataset
from .evaluation import HumanRating
from .experiments import ExperimentRun
from .versioning import PromptDefinition, PromptVersion, VersionStatus

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / ".data" / "prompt_studio.db"


class SQLiteRepository:
    def __init__(self, path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    preferred_version_id TEXT
                );
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    version_id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL REFERENCES prompts(prompt_id),
                    version_number INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('draft','published')),
                    content_hash TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    UNIQUE(prompt_id, version_number, status)
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY(dataset_id, version)
                );
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                    variant TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS human_ratings (
                    result_id TEXT PRIMARY KEY REFERENCES experiment_results(result_id) ON DELETE CASCADE,
                    data_json TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS prevent_published_version_update
                BEFORE UPDATE ON prompt_versions
                WHEN OLD.status = 'published'
                BEGIN SELECT RAISE(ABORT, 'published prompt versions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_published_version_delete
                BEFORE DELETE ON prompt_versions
                WHEN OLD.status = 'published'
                BEGIN SELECT RAISE(ABORT, 'published prompt versions are immutable'); END;
                """
            )

    def save_prompt(self, prompt: PromptDefinition) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO prompts(prompt_id,name,description,created_at,preferred_version_id) VALUES(?,?,?,?,?)",
                (
                    prompt.prompt_id,
                    prompt.name,
                    prompt.description,
                    prompt.created_at.isoformat(),
                    prompt.preferred_version_id,
                ),
            )

    def get_prompt(self, prompt_id: str) -> PromptDefinition | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM prompts WHERE prompt_id=?", (prompt_id,)
            ).fetchone()
        return self._prompt_from_row(row) if row else None

    def list_prompts(self) -> list[PromptDefinition]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM prompts ORDER BY created_at").fetchall()
        return [self._prompt_from_row(row) for row in rows]

    @staticmethod
    def _prompt_from_row(row: sqlite3.Row) -> PromptDefinition:
        return PromptDefinition.model_validate(dict(row))

    def save_version(self, version: PromptVersion) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO prompt_versions(version_id,prompt_id,version_number,status,content_hash,data_json) VALUES(?,?,?,?,?,?)",
                (
                    version.version_id,
                    version.prompt_id,
                    version.version_number,
                    version.status.value,
                    version.content_hash,
                    version.model_dump_json(),
                ),
            )

    def update_draft(self, version: PromptVersion) -> None:
        if version.status is not VersionStatus.DRAFT:
            raise ValueError("only drafts can be updated")
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE prompt_versions SET content_hash=?, data_json=? WHERE version_id=? AND status='draft'",
                (version.content_hash, version.model_dump_json(), version.version_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("draft was not found or is immutable")

    def publish_draft(self, draft_id: str, published: PromptVersion) -> None:
        """Atomically replace an editable draft with its immutable snapshot."""

        with self.connect() as connection:
            draft = connection.execute(
                "SELECT status FROM prompt_versions WHERE version_id=?", (draft_id,)
            ).fetchone()
            if draft is None or draft["status"] != "draft":
                raise ValueError("draft was not found or is immutable")
            connection.execute(
                "INSERT INTO prompt_versions(version_id,prompt_id,version_number,status,content_hash,data_json) VALUES(?,?,?,?,?,?)",
                (
                    published.version_id,
                    published.prompt_id,
                    published.version_number,
                    published.status.value,
                    published.content_hash,
                    published.model_dump_json(),
                ),
            )
            connection.execute(
                "DELETE FROM prompt_versions WHERE version_id=? AND status='draft'",
                (draft_id,),
            )

    def get_version(self, version_id: str) -> PromptVersion | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM prompt_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        return PromptVersion.model_validate_json(row["data_json"]) if row else None

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT data_json FROM prompt_versions WHERE prompt_id=? ORDER BY version_number, status",
                (prompt_id,),
            ).fetchall()
        return [PromptVersion.model_validate_json(row["data_json"]) for row in rows]

    def set_preferred_version(self, prompt_id: str, version_id: str) -> None:
        version = self.get_version(version_id)
        if (
            version is None
            or version.prompt_id != prompt_id
            or version.status is not VersionStatus.PUBLISHED
        ):
            raise ValueError("preferred version must be a published version for the prompt")
        with self.connect() as connection:
            connection.execute(
                "UPDATE prompts SET preferred_version_id=? WHERE prompt_id=?",
                (version_id, prompt_id),
            )

    def save_dataset(self, dataset: EvaluationDataset) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO datasets(dataset_id,version,data_json) VALUES(?,?,?)",
                (dataset.dataset_id, dataset.version, dataset.model_dump_json()),
            )

    def get_dataset(self, dataset_id: str, version: str) -> EvaluationDataset | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM datasets WHERE dataset_id=? AND version=?",
                (dataset_id, version),
            ).fetchone()
        return EvaluationDataset.model_validate_json(row["data_json"]) if row else None

    def list_datasets(self) -> list[EvaluationDataset]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT data_json FROM datasets ORDER BY dataset_id,version"
            ).fetchall()
        return [EvaluationDataset.model_validate_json(row["data_json"]) for row in rows]

    def save_run(self, run: ExperimentRun) -> bool:
        """Idempotently save one run and all case results in one transaction."""

        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO experiment_runs(run_id,experiment_id,created_at,status,data_json) VALUES(?,?,?,?,?)",
                (
                    run.run_id,
                    run.plan.experiment_id,
                    run.completed_at.isoformat(),
                    run.status,
                    run.model_dump_json(),
                ),
            )
            if cursor.rowcount == 0:
                return False
            connection.executemany(
                "INSERT INTO experiment_results(result_id,run_id,variant,case_id,status,data_json) VALUES(?,?,?,?,?,?)",
                [
                    (
                        result.result_id,
                        run.run_id,
                        result.variant,
                        result.case_id,
                        result.status,
                        result.model_dump_json(),
                    )
                    for result in run.results
                ],
            )
        return True

    def get_run(self, run_id: str) -> ExperimentRun | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM experiment_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return ExperimentRun.model_validate_json(row["data_json"]) if row else None

    def list_runs(self) -> list[ExperimentRun]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT data_json FROM experiment_runs ORDER BY created_at DESC"
            ).fetchall()
        return [ExperimentRun.model_validate_json(row["data_json"]) for row in rows]

    def delete_run(self, run_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM experiment_runs WHERE run_id=?", (run_id,))

    def save_rating(self, rating: HumanRating) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO human_ratings(result_id,data_json) VALUES(?,?) ON CONFLICT(result_id) DO UPDATE SET data_json=excluded.data_json",
                (rating.result_id, rating.model_dump_json()),
            )

    def get_ratings(self, run_id: str) -> dict[str, HumanRating]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT h.data_json FROM human_ratings h JOIN experiment_results r ON r.result_id=h.result_id WHERE r.run_id=?",
                (run_id,),
            ).fetchall()
        ratings = [HumanRating.model_validate_json(row["data_json"]) for row in rows]
        return {rating.result_id: rating for rating in ratings}
