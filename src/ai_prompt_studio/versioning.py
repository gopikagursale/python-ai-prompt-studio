"""Local immutable prompt version models and management."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .exceptions import PromptValidationError


def utc_now() -> datetime:
    return datetime.now(UTC)


class VersionStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"


class InputVariableDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    description: str = ""
    required: bool = True


class PromptDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_id: str
    name: str = Field(min_length=1)
    description: str
    created_at: datetime
    preferred_version_id: str | None = None


class PromptVersion(BaseModel):
    """Immutable prompt snapshot; drafts are replaced, published rows never mutate."""

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    version_id: str
    version_number: int = Field(ge=1)
    prompt_name: str = Field(min_length=1)
    description: str
    prompt_template: str = Field(min_length=1)
    developer_instructions: str = Field(min_length=1)
    technique_settings: dict[str, Any] = Field(default_factory=dict)
    input_variables: tuple[InputVariableDefinition, ...] = ()
    output_constraints: tuple[str, ...] = ()
    structured_output_schema_id: str | None = None
    created_at: datetime
    change_description: str = Field(min_length=1)
    status: VersionStatus
    content_hash: str

    @field_validator("prompt_template", "developer_instructions", "change_description")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value


def prompt_content_hash(data: dict[str, Any]) -> str:
    content = {
        key: data[key]
        for key in (
            "prompt_name",
            "description",
            "prompt_template",
            "developer_instructions",
            "technique_settings",
            "input_variables",
            "output_constraints",
            "structured_output_schema_id",
        )
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class VersionRepository(Protocol):
    def save_prompt(self, prompt: PromptDefinition) -> None: ...
    def get_prompt(self, prompt_id: str) -> PromptDefinition | None: ...
    def list_prompts(self) -> list[PromptDefinition]: ...
    def save_version(self, version: PromptVersion) -> None: ...
    def publish_draft(self, draft_id: str, published: PromptVersion) -> None: ...
    def update_draft(self, version: PromptVersion) -> None: ...
    def get_version(self, version_id: str) -> PromptVersion | None: ...
    def list_versions(self, prompt_id: str) -> list[PromptVersion]: ...
    def set_preferred_version(self, prompt_id: str, version_id: str) -> None: ...


class PromptVersionManager:
    def __init__(self, repository: VersionRepository) -> None:
        self.repository = repository

    def create_prompt(self, name: str, description: str = "") -> PromptDefinition:
        if not name.strip():
            raise PromptValidationError("Prompt name cannot be empty.")
        prompt = PromptDefinition(
            prompt_id=f"prompt_{uuid4().hex}",
            name=name,
            description=description,
            created_at=utc_now(),
        )
        self.repository.save_prompt(prompt)
        return prompt

    def create_initial_version(
        self,
        prompt: PromptDefinition,
        *,
        prompt_template: str,
        developer_instructions: str,
        input_variables: tuple[InputVariableDefinition, ...] = (),
        technique_settings: dict[str, Any] | None = None,
        output_constraints: tuple[str, ...] = (),
        structured_output_schema_id: str | None = None,
        change_description: str = "Initial version",
    ) -> PromptVersion:
        if self.repository.list_versions(prompt.prompt_id):
            raise PromptValidationError("This prompt already has an initial version.")
        version = self._make_version(
            prompt,
            version_number=1,
            status=VersionStatus.PUBLISHED,
            prompt_template=prompt_template,
            developer_instructions=developer_instructions,
            input_variables=input_variables,
            technique_settings=technique_settings or {},
            output_constraints=output_constraints,
            structured_output_schema_id=structured_output_schema_id,
            change_description=change_description,
        )
        self.repository.save_version(version)
        self.repository.set_preferred_version(prompt.prompt_id, version.version_id)
        return version

    def create_draft_from_version(self, version_id: str) -> PromptVersion:
        source = self.require_version(version_id)
        prompt = self.require_prompt(source.prompt_id)
        next_number = (
            max(
                (
                    version.version_number
                    for version in self.repository.list_versions(prompt.prompt_id)
                ),
                default=0,
            )
            + 1
        )
        draft = self._make_version(
            prompt,
            version_number=next_number,
            status=VersionStatus.DRAFT,
            prompt_template=source.prompt_template,
            developer_instructions=source.developer_instructions,
            input_variables=source.input_variables,
            technique_settings=source.technique_settings,
            output_constraints=source.output_constraints,
            structured_output_schema_id=source.structured_output_schema_id,
            change_description=f"Draft copied from v{source.version_number}",
        )
        self.repository.save_version(draft)
        return draft

    def edit_draft(self, version_id: str, **changes: Any) -> PromptVersion:
        current = self.require_version(version_id)
        if current.status is not VersionStatus.DRAFT:
            raise PromptValidationError(
                "Published versions are immutable; create a new draft instead."
            )
        allowed = {
            "prompt_name",
            "description",
            "prompt_template",
            "developer_instructions",
            "technique_settings",
            "input_variables",
            "output_constraints",
            "structured_output_schema_id",
            "change_description",
        }
        if unknown := set(changes) - allowed:
            raise PromptValidationError(f"Unsupported draft fields: {', '.join(sorted(unknown))}.")
        data = current.model_dump()
        data.update(changes)
        data["content_hash"] = prompt_content_hash(data)
        updated = PromptVersion.model_validate(data)
        self.repository.update_draft(updated)
        return updated

    def publish_draft(self, version_id: str) -> PromptVersion:
        draft = self.require_version(version_id)
        if draft.status is not VersionStatus.DRAFT:
            raise PromptValidationError("Only drafts can be published.")
        data = draft.model_dump()
        data.update(
            version_id=f"version_{uuid4().hex}",
            status=VersionStatus.PUBLISHED,
            created_at=utc_now(),
        )
        data["content_hash"] = prompt_content_hash(data)
        published = PromptVersion.model_validate(data)
        self.repository.publish_draft(draft.version_id, published)
        return published

    def mark_preferred(self, version_id: str) -> None:
        version = self.require_version(version_id)
        if version.status is not VersionStatus.PUBLISHED:
            raise PromptValidationError("Only published versions can be preferred.")
        self.repository.set_preferred_version(version.prompt_id, version.version_id)

    def compare_versions(self, first_id: str, second_id: str) -> dict[str, tuple[Any, Any]]:
        first, second = self.require_version(first_id), self.require_version(second_id)
        fields = (
            "prompt_template",
            "developer_instructions",
            "technique_settings",
            "input_variables",
            "output_constraints",
            "structured_output_schema_id",
            "content_hash",
        )
        return {
            field: (getattr(first, field), getattr(second, field))
            for field in fields
            if getattr(first, field) != getattr(second, field)
        }

    def require_prompt(self, prompt_id: str) -> PromptDefinition:
        prompt = self.repository.get_prompt(prompt_id)
        if prompt is None:
            raise PromptValidationError("Prompt not found.")
        return prompt

    def require_version(self, version_id: str) -> PromptVersion:
        version = self.repository.get_version(version_id)
        if version is None:
            raise PromptValidationError("Prompt version not found.")
        return version

    @staticmethod
    def _make_version(
        prompt: PromptDefinition,
        *,
        version_number: int,
        status: VersionStatus,
        **values: Any,
    ) -> PromptVersion:
        data = {
            "prompt_id": prompt.prompt_id,
            "version_id": f"version_{uuid4().hex}",
            "version_number": version_number,
            "prompt_name": prompt.name,
            "description": prompt.description,
            "created_at": utc_now(),
            "status": status,
            **values,
        }
        data["content_hash"] = prompt_content_hash(data)
        return PromptVersion.model_validate(data)
