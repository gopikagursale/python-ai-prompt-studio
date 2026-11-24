# Project Plan

## Milestone 1 — implemented

- establish the `src` package layout and environment configuration
- model and validate reusable prompt templates
- add beginner-explanation and passage-summary templates
- build prompts with a clear instruction/content boundary
- integrate the OpenAI Responses API for stateless requests
- add an interactive preview-and-confirm CLI
- cover rendering, validation, API arguments, results, and safe failures with tests
- document setup, architecture, and prompt-engineering concepts

## Milestone 2 — implemented

- add the official Streamlit dependency and root entry point
- generate input widgets from the existing template registry
- reuse `PromptBuilder` validation for local previews
- add editable user-prompt confirmation while preserving instruction separation
- reuse `OpenAIModelClient` for the only paid-call boundary
- display response text, configured model, template metadata, token usage, status, and latency
- manage template, preview, confirmation, response, clear, and reset state safely
- prevent stale answers and duplicate calls on ordinary Streamlit reruns
- add workflow unit tests and Streamlit AppTest coverage
- preserve the Milestone 1 CLI and all existing tests

## Milestone 3 — implemented

- add typed zero-shot, few-shot, role-based, and constraint-based techniques
- add validated 1–5 example demonstrations and a sentiment-classification default set
- add four application-approved communication personas
- preserve user-controlled examples and constraint values in user content
- add exact technique previews and explicit confirmation/execution gates
- add deterministic word, bullet, keyword, and forbidden-phrase checks
- add zero-shot versus modified-technique comparison snapshots
- execute comparison variants independently using the same task and model
- preserve all M1/M2 workflows and expand mocked unit/AppTest coverage

## Milestone 4 — implemented

- add strict Pydantic schemas for sentiment, support tickets, and article summaries
- add an approved-schema registry and safe schema inspection
- generate required-field JSON Schema with no additional properties
- extend the existing client with `responses.parse` and `text_format`
- keep untrusted extraction source separate from developer instructions
- handle refusals, incomplete/truncated output, missing parsed data, unsupported requests, and validation failures
- display formatted JSON, typed Pydantic instances, usage, latency, and layered validation
- mark support-ticket priority as a suggestion requiring verification
- invalidate stale structured previews/results without duplicate requests
- preserve all M1–M3 workflows and expand mocked unit/AppTest coverage

## Milestone 5 — implemented

- add local prompt definitions, editable drafts, immutable published versions, preferred-version selection, and content hashes
- add a version-controlled 12-case synthetic sentiment dataset and size/schema-validated JSON import
- construct auditable paired A/B plans that hold model, dataset, trial count, generation behavior, and case inputs constant
- preview the exact request count and require explicit confirmation before sequential execution
- preserve per-case successes and safe failures, usage, latency, actual API-attempt count, and effective request snapshots
- evaluate exact labels, JSON/schema properties, required fields, enums, constraints, and nonempty output deterministically
- add separate optional 1–5 human dimensions without an unexplained aggregate quality score
- calculate planned-denominator and completed-only accuracy, format/constraint rates, tokens, latency, and paired regressions
- persist opt-in history and ratings in local SQLite with transactional, idempotent saves
- add safe CSV/JSON exports and Streamlit analytics/history controls
- preserve all M1–M4 workflows and expand unit/AppTest coverage without live API calls

## Milestone 6 — implemented

- add Ruff lint/format configuration and resolve existing findings/deprecations
- strengthen security regression tests and generated-data ignore rules
- add read-only GitHub Actions CI for Python 3.11, Ruff, and pytest
- produce a portfolio README and consolidated lifecycle architecture
- add user, manual-test, and release checklists
- inspect Projects 1 and 2 read-only and create a verified three-project study guide
- add 60 answered interview questions and a realistic 14-day revision plan
- retain mocked external APIs and avoid automatic deployment, commits, or pushes
- document the missing Git repository/history and local credential as release blockers

## Deferred scope

The following are intentionally deferred to later milestones or projects:

- persistent few-shot example library
- user-defined or dynamically imported output schemas
- production traffic A/B splitting and statistical significance testing
- cloud or multi-user prompt history
- retrieval-augmented generation (RAG)
- agents
- LangChain
- multi-provider comparison

## Suggested next milestone

Expand evaluation coverage with task-specific datasets and independently
reviewed rubrics before considering production use. Statistical testing,
LLM-as-a-judge, background scheduling, and production traffic splitting remain
explicitly outside this local learning project.

