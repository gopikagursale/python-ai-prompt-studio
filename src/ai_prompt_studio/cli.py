"""Interactive command-line interface."""

from __future__ import annotations

from collections.abc import Callable

from .builder import PromptBuilder
from .client import OpenAIModelClient
from .config import AppSettings
from .exceptions import PromptStudioError
from .models import GenerationResult, PromptTemplate, RenderedPrompt
from .templates import TEMPLATES

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _choose_template(input_fn: InputFn, output_fn: OutputFn) -> PromptTemplate | None:
    output_fn("\nAvailable templates:")
    for index, template in enumerate(TEMPLATES, start=1):
        output_fn(f"  {index}. {template.name} (v{template.version}, id: {template.template_id})")
    output_fn("  q. Exit")

    while True:
        choice = input_fn("Choose a template: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(TEMPLATES):
            return TEMPLATES[int(choice) - 1]
        output_fn("Please enter a listed number or q to exit.")


def _collect_variables(template: PromptTemplate, input_fn: InputFn) -> dict[str, str]:
    values: dict[str, str] = {}
    for variable in template.required_input_variables:
        while True:
            value = input_fn(f"Enter {variable}: ")
            if value.strip():
                values[variable] = value
                break
            print(f"{variable} cannot be empty.")
    return values


def _show_preview(prompt: RenderedPrompt, model: str, output_fn: OutputFn = print) -> None:
    output_fn("\n--- Prompt preview (no API request has been made) ---")
    output_fn(f"Template: {prompt.template_name}")
    output_fn(f"Version: {prompt.template_version}")
    output_fn(f"Selected model: {model}")
    output_fn("\nDeveloper instructions:")
    output_fn(prompt.developer_instructions)
    output_fn("\nRendered user prompt:")
    output_fn(prompt.user_prompt)
    output_fn("--- End preview ---")


def _show_result(result: GenerationResult, output_fn: OutputFn = print) -> None:
    output_fn("\nModel response:")
    output_fn(result.text)
    if result.usage:
        output_fn("\nToken usage:")
        output_fn(f"  Input: {result.usage.input_tokens}")
        output_fn(f"  Output: {result.usage.output_tokens}")
        output_fn(f"  Total: {result.usage.total_tokens}")
    else:
        output_fn("\nToken usage: not provided by the API.")


def run(input_fn: InputFn = input, output_fn: OutputFn = print) -> int:
    """Run the prompt studio until the user exits."""

    output_fn("AI Prompt Engineering & Content Studio")
    try:
        settings = AppSettings.from_env()
    except PromptStudioError as exc:
        output_fn(f"Configuration error: {exc}")
        output_fn("Copy .env.example to .env and add your own settings.")
        return 2

    builder = PromptBuilder(settings.developer_instructions)
    client = OpenAIModelClient(settings)

    while True:
        template = _choose_template(input_fn, output_fn)
        if template is None:
            output_fn("Goodbye.")
            return 0

        try:
            variables = _collect_variables(template, input_fn)
            prompt = builder.build(template, variables)
            _show_preview(prompt, settings.model, output_fn)
            confirm = input_fn("\nRun this prompt? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                output_fn("Request cancelled; no API call was made.")
                continue
            result = client.generate(prompt)
            _show_result(result, output_fn)
        except PromptStudioError as exc:
            output_fn(f"Error: {exc}")

        again = input_fn("\nCreate another prompt? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            output_fn("Goodbye.")
            return 0


def main() -> None:
    """Console script entry point."""

    try:
        raise SystemExit(run())
    except (EOFError, KeyboardInterrupt):
        print("\nExited without sending another request.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
