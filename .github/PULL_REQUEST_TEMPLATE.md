<!--
Thanks for contributing to Polaris. Please keep changes focused and well-scoped.
Large architectural changes must be discussed in an issue first (see CONTRIBUTING.md).
-->

## Summary

<!-- What does this PR change, and why? -->

## Related issue

<!-- e.g. "Closes #123", or "N/A". -->

## Type of change

- [ ] Bug fix
- [ ] New functionality (vertical slice — leaves Polaris runnable)
- [ ] Documentation
- [ ] Refactor / internal cleanup
- [ ] Tooling / CI

## Checklist

- [ ] The change is focused and scoped to a single concern.
- [ ] It follows the project's engineering philosophy — concrete before abstract,
      no speculative infrastructure, abstractions justified by ≥2 implementations
      (see `CONTRIBUTING.md` and `docs/adr/`).
- [ ] Any significant architectural decision is captured in an ADR (`docs/adr/`).
- [ ] Tests added or updated, and they run **offline** (no network/downloads).
- [ ] Documentation updated where relevant (module README, `docs/design/`, `CHANGELOG.md`).
- [ ] `uv run black --check .` passes.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run mypy polaris` passes.
- [ ] `uv run pytest` passes.
