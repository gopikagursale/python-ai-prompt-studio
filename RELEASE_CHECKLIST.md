# Release Checklist

## Automated quality

- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m pytest`
- [ ] Headless Streamlit startup succeeds.
- [ ] CI passes on a real GitHub push/PR without API secrets.

## Security and privacy

- [ ] Rotate the current local development API key before publication.
- [ ] Initialize/locate Git metadata and confirm the intended `main` branch/remote.
- [ ] Run a Git-history secret scanner; `.gitignore` is not historical cleanup.
- [ ] Confirm `.env`, Streamlit secrets, databases/journals, exports, and caches are untracked.
- [ ] Confirm `.env.example` contains placeholders only.
- [ ] Inspect every tracked JSON/CSV/database-like artifact for private data.
- [ ] Delete or securely retain local experiment databases outside the public tree.
- [ ] Verify screenshots contain no keys, customer text, account IDs, or private URLs.
- [ ] Review dependency advisories and repository secret-scanning settings.

If any real credential exists in history: revoke/rotate it immediately, do not
publish, and plan reviewed history remediation separately. Never force-push or
rewrite history automatically as part of this checklist.

## Documentation and portfolio claims

- [ ] README commands work from a clean clone on Python 3.11.
- [ ] Architecture diagram renders on GitHub.
- [ ] User/manual guides match current widget labels.
- [ ] No claim says production-ready, statistically significant, or live-verified without evidence.
- [ ] Repository description identifies this as a local learning/portfolio project.
- [ ] License and author/contact choices are reviewed by the owner.

## Human QA

- [ ] Complete every free item in `MANUAL_TEST_CHECKLIST.md`.
- [ ] Decide whether to perform optional paid smoke tests.
- [ ] Verify accessibility basics: labels, keyboard flow, contrast, narrow viewport.
- [ ] Verify destructive deletion confirmation and local backup expectations.
- [ ] Review saved-data privacy and retention expectations.

## Release decision

Release only when automated checks pass, Git history is available and scanned,
the credential has been rotated, manual QA is recorded, and the owner accepts
the local single-user/security limitations.
