from __future__ import annotations

from unittest.mock import Mock

from ai_prompt_studio.cli import _show_preview, run
from ai_prompt_studio.config import AppSettings
from ai_prompt_studio.models import RenderedPrompt


def test_preview_contains_required_information() -> None:
    lines: list[str] = []
    prompt = RenderedPrompt(
        template_id="example",
        template_name="Example template",
        template_version="2.1.0",
        developer_instructions="Developer guidance",
        user_prompt="Rendered user content",
    )

    _show_preview(prompt, "selected-model", lines.append)
    preview = "\n".join(lines)

    assert "Example template" in preview
    assert "2.1.0" in preview
    assert "Developer guidance" in preview
    assert "Rendered user content" in preview
    assert "selected-model" in preview


def test_declining_confirmation_makes_no_api_request(monkeypatch) -> None:
    settings = AppSettings(
        api_key="unused-test-key",
        model="test-model",
        developer_instructions="Trusted instructions.",
    )
    client = Mock()
    monkeypatch.setattr("ai_prompt_studio.cli.AppSettings.from_env", lambda: settings)
    monkeypatch.setattr("ai_prompt_studio.cli.OpenAIModelClient", lambda unused_settings: client)
    answers = iter(["1", "recursion", "n", "q"])

    exit_code = run(input_fn=lambda unused_prompt: next(answers), output_fn=lambda _: None)

    assert exit_code == 0
    client.generate.assert_not_called()
