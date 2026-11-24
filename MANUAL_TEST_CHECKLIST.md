# Manual Test Checklist

Automated tests verify Python behavior with mocks. The checks below validate
real browser behavior and, where clearly marked, optional live provider access.
Do not perform paid checks without accepting the cost.

## Free local checks

- [ ] Fresh editable installation succeeds on Python 3.11.
- [ ] Missing `.env` shows a safe configuration message without a credential.
- [ ] Streamlit starts and all four workspaces render responsively.
- [ ] Template selection changes fields without stale values or widget errors.
- [ ] Empty/missing variables are rejected locally.
- [ ] Braces, Unicode, quotes, and multiline user values preview literally.
- [ ] Preview shows developer instructions separately and makes no request.
- [ ] Editing after confirmation disables Run and removes stale output.
- [ ] Clear Response and Reset Prompt Form behave independently.
- [ ] Few-shot editor enforces 1–5 nonempty pairs and preserves order.
- [ ] Persona selection displays only approved personas.
- [ ] Constraint preview and deterministic checks explain their limits.
- [ ] Structured schema/source changes invalidate old confirmation/result.
- [ ] JSON Schema viewer matches the selected Pydantic model.
- [ ] Prompt version compare, draft save, publish, restore, and preferred controls work.
- [ ] Published versions cannot be edited or deleted through normal controls.
- [ ] Dataset preview contains 12 unique synthetic cases.
- [ ] Invalid/oversized/extra-field dataset JSON is rejected.
- [ ] Experiment plan shows `cases × 2 × trials` and exact request snapshots.
- [ ] Run remains disabled before plan confirmation.
- [ ] Save Results defaults off.
- [ ] Saved run opens without changing API-attempt counters.
- [ ] Blinded rating hides variant labels and saves without inference.
- [ ] Confirmed deletion removes the run; cancellation preserves it.
- [ ] CSV/JSON downloads contain IDs, versions, outcomes, usage, and latency.
- [ ] CSV text beginning `=`, `+`, `-`, or `@` is neutralized.
- [ ] CLI preview defaults to not running when Enter is pressed.

## Optional paid live checks

- [ ] Rotate/create a dedicated low-risk development key with account limits.
- [ ] Run one template prompt and verify readable text.
- [ ] Verify reported token counts against the provider response when available.
- [ ] Verify authentication, model, timeout, and rate-limit messages with safe test conditions.
- [ ] Run one compatible Structured Output request; verify parsed and raw JSON.
- [ ] Accept the displayed request count before a very small A/B run.
- [ ] Verify actual attempts, partial failure behavior, latency, and optional persistence.

Record date, model ID, tester, environment, and observed results. Do not paste
keys or sensitive prompts into screenshots or issue reports.
