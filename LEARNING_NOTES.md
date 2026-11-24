# Learning Notes

## What prompt engineering means

Prompt engineering is the deliberate design, organization, testing, and
maintenance of the context given to a language model. It is more than clever
wording: it includes instruction priority, data boundaries, constraints,
examples, versioning, and evaluation against representative tasks.

## Prompt versus prompt template

A **prompt** is the concrete content sent for one request. A **prompt template**
is a reusable, versioned definition containing fixed guidance and named slots
for changing values. For example, `Topic to explain: {topic}` is a template;
after substitution, `Topic to explain: neural networks` is one rendered prompt.

## Developer instructions versus user input

Developer instructions express application-owned behavior: the task, rules,
and output constraints. User input is untrusted task data, such as the topic or
passage. This project never concatenates a user value into developer guidance.
It sends the two layers separately as `instructions` and `input` to preserve
their intended priority and make the trust boundary visible in code and preview.

Separation is an important defense, not a mathematical security guarantee. A
model can still be influenced by hostile text, so sensitive actions need
additional validation and authorization outside the prompt.

## Why prompt variables must be validated

Validation catches incomplete requests early, prevents callers from silently
supplying misspelled or unused fields, and guarantees that a template's declared
contract matches its placeholders. Restricting placeholders to simple names
also prevents Python attribute lookup, indexing, conversions, or formatting
features from becoming an unintended mini-language.

User values themselves may contain braces, quotes, Unicode, markup, or other
special characters. They are substituted once as ordinary strings and are not
evaluated as Python or reparsed as template syntax.

## What zero-shot prompting means

Zero-shot prompting asks a model to perform a task from instructions alone,
without showing completed input/output examples. Both Milestone 1 templates are
zero-shot. This keeps them simple, though some tasks become more consistent with
carefully selected few-shot examples—a feature intentionally deferred here.

## Why prompts should be version-controlled

Prompt changes are behavior changes. Version control records who changed a
template, what changed, and why. A template version such as `1.0.0` also lets
logs and evaluations identify the exact wording responsible for an output,
making regressions reproducible.

## Why prompt quality must be evaluated

Fluent output is not automatically correct output. Prompt quality should be
measured against representative cases and explicit criteria such as factuality,
instruction following, completeness, clarity, safety, latency, and cost. Tests
in this milestone verify software behavior, not semantic answer quality. A later
evaluation suite should use fixed datasets, rubrics, and human review where
judgment is required.

## How the prompt builder works

1. A `PromptTemplate` validates its metadata and placeholder contract when it is created.
2. `PromptBuilder.build` compares supplied keys with required variable names.
3. It rejects missing, extra, non-text, and empty values.
4. It substitutes values into `user_prompt_template` exactly once.
5. It combines base guidance, task description, template instructions, and output constraints into developer instructions.
6. It returns an immutable `RenderedPrompt` whose developer and user fields remain separate.

## How the OpenAI integration works

`AppSettings` loads the API key and model ID from environment variables. The
model ID is not hardcoded. It also validates a request timeout and a small retry
limit. `OpenAIModelClient` creates the official SDK client with those settings
and calls `client.responses.create` with:

- `model`: the environment-selected model
- `instructions`: trusted developer guidance
- `input`: rendered user task content
- `store=False`: no application request state is intentionally retained by this call

The adapter extracts `output_text` and token counts when the API supplies them.
It translates expected SDK failures into short, fixed messages instead of
showing raw exception details that could reveal secrets. Tests inject a mocked
SDK client, so automated testing cannot create paid requests.

## What Streamlit is

Streamlit is a Python framework for building interactive data and AI web apps.
Widgets are declared with Python calls. When a user interacts with a widget,
Streamlit normally reruns the script from top to bottom and reconstructs the UI
from widget and session state.

## How the UI communicates with the prompt builder

The Streamlit page reads a template from the existing registry and creates one
input widget per required variable. On **Preview Prompt**, it sends the selected
template ID and collected values through `studio.build_preview` to the existing
`PromptBuilder`. The page does not reproduce validation or rendering rules.

## Why business logic remains separate from UI

UI code changes frequently and is tied to Streamlit's rerun model. Prompt
validation and API behavior should also work from the CLI and tests. Keeping
those rules in package modules prevents duplication, makes failures easier to
test, and allows a future UI to reuse the same trusted behavior.

## How templates become input forms

Every `PromptTemplate` declares `required_input_variables`. The page loops over
that tuple and creates a text widget with a stable key containing both template
ID and variable name. The summary passage receives a multiline field; shorter
variables receive a single-line field. No UI-only variable is invented.

## What prompt preview means

A preview is a locally rendered `RenderedPrompt`. It shows template metadata,
configured model, developer instructions, and user prompt before execution.
The preview calls only local Python validation and substitution, so it consumes
no model tokens and cannot create an API charge.

## Developer instructions and editable user input

Developer instructions remain read-only and travel through the Responses API
`instructions` argument. Only the user-prompt field is editable. Confirmation
creates a new immutable prompt with the exact visible user text while preserving
the original instructions. Editing again invalidates that confirmation and any
older displayed answer.

## How `st.form` works

A Streamlit form groups multiple widgets. Changes are submitted together when
the form's submit button is clicked. Here that button is **Preview Prompt**.
This gives the builder a coherent set of values and provides a clear local-only
stage before confirmation and execution.

## How `st.session_state` works

`st.session_state` persists values across reruns for one browser session. This
app keeps only template selection, template-scoped fields, preview, confirmed
prompt, execution result, usage, latency, and status. The API key stays in
environment-backed settings and is never copied into session state.

Widget-owned keys must not be assigned after their widgets are instantiated in
the same run. The app initializes values before widget construction and uses
supported callbacks for template change, editor change, clear, and reset.

## Why reruns can cause mistakes

If API code runs unconditionally, changing an unrelated widget could repeat a
paid request. Stale state can also display an old answer beside a new prompt.
The app prevents this by placing the model call only inside the explicit Run
button branch and by invalidating derived state when templates, previews, or
confirmed editor content change. On ordinary reruns, the button is false and
the stored result is only displayed—not executed again.

## UI history is not LLM conversation context

Session state is a browser-session convenience, not model memory. The latest
result remains visible for the user, but it is not included in the next API
request. Every request continues to use `store=False` and contains only the
current developer instructions and confirmed user prompt.

## Run the Streamlit app and tests

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Run all CLI, core, workflow, and Streamlit tests with:

```bash
pytest
```

AppTest simulates widgets and reruns with a mocked model method. It verifies UI
behavior without starting a browser server or contacting OpenAI.

## Zero-shot prompting

Zero-shot prompting describes the task without providing completed examples.
For example:

```text
Classify this review as Positive, Negative, or Neutral:
"The delivery was late, but the product is excellent."
```

It is simple and token-efficient, but the model must infer the desired decision
boundary and output style from instructions alone. The lab verifies that a
zero-shot prompt contains no demonstrations.

## Few-shot prompting

Few-shot prompting places completed example input/output pairs before the new
task. Examples can clarify labels, tone, format, and edge-case behavior. The lab
accepts one to five pairs, preserves their order, labels each demonstration, and
separates them from the final task with an explicit boundary.

Examples influence a model through its current context. They do not update model
weights and therefore are not fine-tuning or training. User-supplied examples
can also contain hostile instructions, so the entire demonstration block stays
in user content rather than being promoted into developer instructions.

## Role-based prompting

A role-based prompt asks the model to communicate according to a persona such as
a primary-school teacher or professional editor. In this project personas come
from an application-controlled registry with a name, description, style, and
scope. Arbitrary user text cannot create a privileged persona.

A persona is not an OpenAI API message role. “Technical instructor” describes
behavior. `instructions` and `input` define the technical trust boundary used by
the API. Approved persona behavior is added to developer instructions, while the
user's task remains user content—even if it contains words such as “system” or
“developer.”

## Constraint-based prompting

Constraint-based prompts state desired output properties: maximum words, tone,
exact bullet count, required keywords, forbidden phrases, or an example request.
Pydantic performs hard validation on configuration values before a request.
The constraints themselves are still instructions to a probabilistic model, so
the model might not follow them.

After execution, Python can check some visible properties deterministically:

- maximum word count;
- exact bullet-line count;
- case-insensitive presence of required keywords;
- case-insensitive absence of forbidden phrases.

Tone and whether an example is genuinely useful require semantic judgment, so
M3 reports them as **not evaluated**. Words are Unicode word groups with internal
hyphens/apostrophes kept together. Bullet lines begin with `-`, `*`, `•`, or a
numbered marker. A passing check confirms only that narrow formatting property,
not truthfulness or answer quality.

## Inspecting the effective prompt

The Technique Lab preview shows technique, model, developer instructions, final
user content, included demonstrations, and active constraints. This is a local
Pydantic/Python construction step; it is not inference and cannot consume model
tokens. Confirmation stores an immutable snapshot. If a task, technique,
example, persona, or constraint changes, the snapshot and prior result are
invalidated.

## Comparing techniques fairly

The lab pairs a zero-shot baseline with the selected modified technique. Both
snapshots use the same task and model, and the OpenAI client supplies the same
request behavior. Each variant must be run explicitly and is labeled with its
effective prompt. Responses, tokens, and latency are observations—not proof of a
winner. Formal A/B evaluation needs repeated samples, evaluation criteria, and
statistical design and is intentionally deferred.

## Prompt injection and trust boundaries

Prompt injection is untrusted content attempting to change or override the
application's instructions. Examples, tasks, keywords, and forbidden phrases
are all user-controlled and remain in user content. Only fixed application
guidance and registry-approved personas enter developer instructions. This
separation reduces privilege confusion, although it cannot guarantee that every
model response will resist malicious text.

## Structured Outputs

Structured Outputs constrain a completed, non-refusal model response to a
supported JSON Schema. The Python SDK can derive that schema from a Pydantic
`BaseModel` and return `response.output_parsed` as a typed object. Applications
must still handle refusals, incomplete responses, API failures, and business
rules. See the [official OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

Structured Outputs differ from JSON Mode. JSON Mode aims to produce syntactically
valid JSON, but it does not guarantee that required fields, types, and enums
match a particular schema. Structured Outputs add schema adherence when used
with a compatible model and supported schema.

## JSON, JSON Schema, Pydantic, and Python instances

- **JSON data** is a concrete serialized value, such as `{"sentiment":"positive"}`.
- **JSON Schema** describes permitted JSON structure, required fields, types, and allowed enum values.
- **A Pydantic model class** is Python code declaring a typed structure by inheriting from `BaseModel`.
- **A Python model instance** is validated runtime data with named, typed attributes.

M4 demonstrates `model_json_schema()` for inspection, `model_validate()` for
runtime validation, `model_dump()` for Python-compatible dictionaries, and
`model_dump_json()` for serialized JSON.

## The three approved schemas

`SentimentAnalysis` uses a `Literal` value of positive, negative, neutral, or
mixed, plus positive points, negative points, and an explanation.

`CustomerSupportTicket` uses category and priority literals, a summary, and
action items. Priority is a model-generated suggestion; it is never treated as
an operational decision without human or business-rule verification.

`ArticleSummary` contains a title, summary, key points, and keywords. No schema
contains a confidence percentage because an arbitrary model number should not
be presented as a calibrated probability.

All fields are required, and extra fields are forbidden. This produces
`additionalProperties: false`. An explicit registry maps fixed identifiers to
model classes, so user text cannot request arbitrary imports or dynamic code.

## API parsing and validation flow

The lab builds a local preview containing approved developer instructions,
untrusted source text, configured model, and generated schema. Previewing is not
inference. Only the confirmed Generate action calls:

```python
client.responses.parse(
    model=configured_model,
    instructions=preview.developer_instructions,
    input=preview.user_content,
    text_format=SelectedPydanticModel,
    store=False,
)
```

The service verifies completion before reading output, detects refusal content,
requires `output_parsed`, and calls `model_validate()` again. Partial or
token-truncated JSON is rejected rather than repaired or accepted. The
configured model is never silently replaced when Structured Outputs are
unsupported.

## Refusal, incomplete output, and validation failure

A refusal explicitly declines the task; it is not a schema-validation error. An
incomplete response stopped before completion, possibly because of an output
token limit or content filter. Missing parsed output means a completed response
still lacked a typed result. A Pydantic failure means returned data violated
field, type, enum, required-field, or extra-field rules. Each case receives a
separate safe error.

## Structure versus truth and business policy

Schema validation can answer “Is `action_items` a list of strings?” It cannot
answer “Is this summary true?” or “Should this ticket really be high priority?”
Business rules therefore run in a separate layer, while factual or semantic
verification may require source comparison and human review. The UI displays
API adherence, Pydantic validation, business checks, and factual limitations as
separate concepts.

## Supported-schema limitations

Structured Outputs support a subset of JSON Schema and require compatible
models. M4 uses flat required objects, strings, arrays of strings, and literal
enums. It avoids dynamic schemas, unsupported keywords, deep nesting, and
optional fields. Schema compilation can add first-request latency, and valid
structured content must still be treated as untrusted application data.

## Prompt versioning and immutable snapshots

A prompt version is the complete behavior-affecting snapshot used by a request:
template text, developer instructions, input contract, technique settings,
constraints, schema ID, metadata, UTC creation time, and change description.
Published versions are immutable because changing one after an experiment would
destroy reproducibility: the recorded version ID would no longer identify what
was actually tested. An older version is changed by copying it into a new draft
and publishing a new numbered snapshot. A stable SHA-256 hash makes accidental
content differences visible.

Local prompt versioning and Git solve different problems. Git versions source
files and documentation across commits. The SQLite version manager records
prompt artifacts created through the running studio and connects exact prompt
snapshots to datasets and experiment results. Both are useful; neither replaces
the other.

## Controlled local A/B testing

A prompt A/B test compares a baseline (A) with a candidate (B) over the same
reviewed cases. Model ID, dataset version, inputs, trial count, relevant
generation behavior, evaluation rules, and structured schema should remain
constant. Only the intended prompt change should differ. M5 renders and stores
the effective developer instructions, user content, demonstrations,
constraints, and schema ID for every attempt so the comparison is auditable.

Language-model output is nondeterministic, so one run is an observation rather
than a universal verdict. Repeated trials can reveal variability, but they also
increase paid requests. The UI therefore calculates `cases × 2 variants ×
trials`, caps a plan at 100 attempts, and requires confirmation. Twelve cases
with one trial mean 24 sequential requests. Exact monetary cost is not shown
because it cannot be reliable without current model-specific pricing and actual
token usage.

## Why representative datasets matter

An evaluation measures only the selected examples and criteria. A dataset must
cover typical cases, important failures, and ambiguous edges rather than only
easy examples. The included sentiment dataset has 12 synthetic, manually
reviewed positive, negative, neutral, mixed, and edge cases. Open-ended tasks
should not receive invented ground truth; those may need a rubric and human
review. JSON imports are schema-validated, limited to 1 MB and 100 cases, and
parsed strictly as data—never executed as code.

## Deterministic and human evaluation

Deterministic checks answer narrow objective questions: exact reviewed label,
nonempty output, valid JSON, Pydantic schema adherence, required fields, allowed
enums, word limits, and required or forbidden phrases. Free-text sentiment is
accepted only when an anchored parser can extract one unambiguous complete
label. An explanation containing a label is not guessed. Unparseable output is
an error, not a correct result.

Human evaluation covers accuracy, relevance, clarity, and instruction following
on separate 1–5 scales with optional notes. Missing dimensions stay unscored.
Variant identity can be hidden while reviewing to reduce bias. Human scores are
kept separate from deterministic outcomes; M5 does not use an LLM judge or
combine everything into an unexplained quality number.

Structured Outputs make format checks more reliable because JSON Schema and
Pydantic validate fields, types, and enums. They still do not establish factual
or semantic correctness. A classification can be perfectly schema-valid and
wrong relative to the reviewed label.

## Metrics and regression testing

Exact-match accuracy uses the clear denominator `correct / all planned results
for that variant`, so failed calls are not silently removed. Completed-only
accuracy is reported separately. Unknown token values, missing ratings, and
non-applicable checks remain `N/A`, not zero. Format and constraint pass rates
use only their applicable recorded checks; latency averages only measured
completed requests.

Paired regression analysis groups each case/trial as both correct, A-only
correct, B-only correct, or both incorrect. A candidate regression is a case
where A passed and B did not. This points reviewers to concrete failures but
does not prove statistical significance or identify a universal winner.

## SQLite, session state, privacy, and exports

`st.session_state` lasts only for a browser session and coordinates widgets; it
is not durable history. SQLite stores opt-in prompt versions, datasets, runs,
per-case results/evaluations, and ratings locally. Related run data is written
transactionally, and duplicate saves of the same run ID are ignored. API keys
and headers are never stored. Saved prompts and responses may still contain
sensitive content, so the `.data` database is Git-ignored and should be handled
as private local data.

Save Results defaults to off. Opening history, adding ratings, deleting a run,
and exporting CSV/JSON are database-only operations and never regenerate model
responses. CSV cells beginning with spreadsheet formula characters are prefixed
with an apostrophe to reduce formula-injection risk. JSON preserves typed
metadata without secrets.

The evaluation workflow follows the core pattern recommended in the
[OpenAI evaluation best-practices guide](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
define the objective, collect representative data, choose task-specific
metrics, run comparisons, and continue evaluating as prompts change.

## Milestone 5 limitations

This is a local, single-user learning system. It runs sequentially and offers no
background jobs, production traffic splitting, calibrated cost estimates,
statistical significance, LLM-as-a-judge, access control, cloud database, or
automatic redaction. Twelve seed cases are useful for demonstrating the
workflow but too small and narrow for broad product claims. Live OpenAI behavior
also depends on the configured model; automated tests use mocks and therefore
verify software behavior, not model quality.

## Release quality, security, and CI

Ruff linting finds unused imports/variables, import-order problems, and selected
syntax/style defects; its formatter makes layout deterministic. Pytest remains
the behavioral gate. A green formatter is not proof of correctness, and a green
test suite is not proof of production security, so both are required alongside
manual review.

GitHub Actions runs these gates on pushes and pull requests for `main` using
Python 3.11 and read-only repository contents permission. The workflow has no
production key and cannot perform a paid model request because tests replace the
OpenAI boundary with mocks. CI is continuous integration; this project has no
continuous deployment.

Security is layered: secrets come from environment configuration; raw provider
errors are replaced; imported datasets are size/schema checked and never
executed; SQL is parameterized; CSV formula prefixes are neutralized; and local
databases/exports are ignored. These controls do not make the app suitable for
public multi-user hosting. It lacks authentication, tenant isolation, encrypted
storage, retention controls, abuse protection, and operational monitoring.

`.gitignore` prevents new untracked files from being added accidentally, but it
cannot remove a secret from existing commits. A public release therefore needs
an actual Git-history secret scan and credential rotation in addition to source
inspection. Automated tests also cannot replace optional human browser checks
or controlled live API compatibility testing.
