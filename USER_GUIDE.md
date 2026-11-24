# User Guide

## Before you start

Create and activate a virtual environment, install `.[dev]`, copy
`.env.example` to `.env`, and supply your own API key and supported model ID.
Preview, history, rating, and export actions are local. Run/Generate actions may
make paid API requests.

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

## Template Studio

1. Choose **Template Studio** in the sidebar.
2. Select *Explain a topic to a beginner* or *Summarize a passage*.
3. Enter every displayed variable. Fields come from the template registry.
4. Click **Preview Prompt**. This validates and renders locally.
5. Inspect developer instructions separately from the editable user prompt.
6. If you edit the user prompt, click **Confirm Final Prompt** again.
7. Click **Run Prompt** only when ready to make one request.
8. Review the response, tokens when reported, model, and measured latency.
9. **Clear Response** removes output; **Reset Prompt Form** also resets inputs.

## Prompt Technique Lab

### Zero-shot

Select **Zero Shot**, enter the task, preview, confirm, and explicitly run. No
demonstrations are added.

### Few-shot

Select **Few Shot**. Add, edit, or remove one-to-five input/expected-output
pairs. They are user-controlled demonstrations, not training examples or
developer instructions. Preview shows their order and the separate new task.

### Role-based persona

Select **Role Based** and choose one approved persona. The persona changes
communication behavior but does not change the technical API roles.

### Constraints

Select **Constraint Based** and configure supported word, bullet, tone, keyword,
forbidden-phrase, or example instructions. After execution, deterministic checks
cover observable rules only. Passing does not prove truth or tone quality.

The comparison view shows a zero-shot baseline and modified variant. Each Run
button is a separate request.

## Structured Output Lab

1. Select an approved schema: sentiment, support ticket, or article summary.
2. Inspect required fields, enums, description, and generated JSON Schema.
3. Paste source text and click **Preview Structured Request**.
4. Confirm developer/user separation and schema, then confirm the request.
5. Click **Generate Structured Output** once.
6. Review formatted JSON, the typed Pydantic representation, validation layers,
   usage, and latency.

A schema-valid result can still be factually wrong. Support priority is a model
suggestion requiring human/business-rule approval.

## Prompt versions

Open **Prompt Experiments → Prompt Versions**.

- Select a prompt and inspect published/draft rows, hashes, descriptions, and preferred status.
- Compare two published versions side-by-side.
- **Create Draft from Selected** copies history without changing it.
- Edit template, developer guidance, and change description in Draft editor.
- **Save Draft** remains editable; **Publish Draft** creates an immutable snapshot.
- **Mark Selected Preferred** changes the default reference, not historical content.

## Evaluation datasets

The Dataset tab previews the packaged 12-case synthetic sentiment dataset,
expected labels, and checks. To import JSON, select a file below 1 MB matching
the documented Pydantic structure and click **Validate and Save Dataset**.
Unknown fields, invalid values, duplicates, empty cases, and over 100 cases are
rejected. Imported strings are never executed.

## Controlled A/B experiment

1. Choose the prompt, published baseline A, published candidate B, dataset, and
   one-to-three trials.
2. Leave **Save results locally** off unless persistence is intended.
3. Review cases, variants, trials, and planned paid attempts.
4. Click **Prepare Experiment Plan**; no API call occurs.
5. Inspect version hashes and exact A/B request snapshots.
6. Click **Confirm Full Experiment Plan**.
7. Click **Run Confirmed Experiment**. Requests run sequentially A then B for
   each case/trial; individual failures do not erase successes.

## Interpreting results

- **Exact accuracy / planned**: correct results divided by every planned result
  for that variant, including failed requests.
- **Completed-only accuracy**: correct among parseable completed outputs.
- **Format/constraint pass rate**: passes among applicable checks only.
- **Latency/tokens**: measured or provider-reported; unavailable stays N/A.
- **A-only regression**: baseline correct and candidate incorrect.
- **B-only improvement**: candidate correct and baseline incorrect.

These are exploratory small-sample observations, not statistical proof.

## Human ratings

Ratings require a saved run. Open Results, optionally hide variant identity,
select a response, and score any of accuracy, relevance, clarity, and instruction
following from 1–5. Notes are optional unless no score is given. Saving a rating
is local and causes no model request.

## History, deletion, and export

Saved Experiments lists local SQLite runs. **Open Saved Run** reads the database
without inference. Deletion requires a confirmation checkbox and cascades to
results/ratings. Results provide **Download CSV** and **Download JSON**. CSV
protects common spreadsheet formula prefixes. Exports may contain sensitive
prompts and model text; store them appropriately.

## CLI

```bash
python -m ai_prompt_studio
```

Choose a template, enter variables, inspect the preview, and answer `y` only if
you want the request. The CLI preserves template and API-service behavior but
does not expose all Streamlit labs.

## Troubleshooting

- Configuration error: verify both key and model in ignored `.env`.
- Authentication/rate limit/timeout: use the safe UI message and provider dashboard;
  raw exceptions are intentionally hidden.
- Structured request rejected: verify the configured model supports the schema.
- Old result disappeared: changing an input deliberately invalidates stale state.
- No token metrics: the provider did not return them; the app does not invent values.
- Database concern: set `PROMPT_STUDIO_DB_PATH` to an explicit private path.
