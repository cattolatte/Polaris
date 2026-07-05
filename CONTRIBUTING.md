# Contributing to Polaris

Thank you for your interest in contributing to Polaris — an educational,
reference-implementation-first NLP engineering platform. The primary product is
*the codebase itself*: an end-to-end NLP system built from scratch, clean enough
to read as a teaching text and engineered well enough to run a real task
reproducibly.

Polaris is **stable (v1.x)**: the full lifecycle — data, tokenization, collation,
models, training, evaluation, pretraining, inference, deployment — is implemented
and released. Contributions of all sizes are welcome: bug fixes, documentation,
tests, and new functionality that fits the roadmap.

---

## The ground rules (read these first)

Polaris optimizes for **clarity over features**. These principles decide every
tie-break, and pull requests are reviewed against them:

- **Concrete before abstract.** Build one real implementation first. No
  generality "for later".
- **Evidence-driven abstraction.** A protocol, base class, or generic is only
  justified once **two or more** concrete implementations demand it
  ([ADR-0004](docs/adr/0004-abstraction-policy.md)).
- **Vertical slices.** Every change leaves Polaris **runnable** — never a model
  you can't train or a trainer with nothing to train
  ([ADR-0002](docs/adr/0002-vertical-slice-architecture.md)).
- **No speculative infrastructure.** Config systems, hooks, plugins, and
  extension points arrive only when a real component needs them.
- **From scratch on PyTorch primitives.** Model internals (attention, layernorm,
  blocks) are hand-written; the framework supplies autograd and data plumbing
  ([ADR-0003](docs/adr/0003-tensor-framework.md)).
- **Own the interface.** Third-party libraries are implementation details:
  import optional deps lazily, expose only Polaris-native types at module
  boundaries, wrap failures in a `PolarisError` subclass.
- **Readability over cleverness.** The reader is a learner. Prefer the obvious
  implementation.

Major architectural decisions are recorded as ADRs in [`docs/adr/`](docs/adr/) —
significant changes need one. **Large changes must be discussed in a GitHub
Issue before implementation begins.**

---

## Development setup

Polaris requires **Python 3.12+** and uses [`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/cattolatte/Polaris.git
cd Polaris
uv sync --extra dev --extra datasets --extra torch --extra serving
```

### The four gates

CI runs these on every PR (Python 3.12 & 3.13 × Ubuntu & macOS), and all four
must pass — run them locally before opening a PR:

```bash
uv run black --check .    # formatting (line length 88)
uv run ruff check .       # lint + import sort
uv run mypy polaris       # strict type checking
uv run pytest             # tests (coverage gate: ≥90%)
```

---

## Code conventions

- Start every module with `from __future__ import annotations`.
- Full type hints (MyPy `strict`); modern syntax (`type X = ...`, `StrEnum`,
  `X | None`).
- Immutable value objects: `@dataclass(frozen=True, slots=True)`, invariants
  validated in `__post_init__`, mutable inputs wrapped in `MappingProxyType`.
- Module docstrings carry a **"Design Principles"** section stating what the
  module deliberately does *not* do. Public classes get NumPy-style docstrings
  (Parameters / Returns / Raises / Examples).
- Raise from the `PolarisError` hierarchy at public boundaries; chain with
  `from`.

## Test conventions

- **Unit tests run fully offline** — no network, no downloads, no fetched
  weights. Mock external backends with minimal hand-written fakes at the
  boundary only.
- **Test the public contract, not internals.** For model/training code, assert
  shapes, invariants, and learning behavior on tiny fixtures — not exact floats.
- Function-based tests (no test classes), one behavior per test, a docstring per
  test, grouped with `# ---` banner comments. See
  `tests/unit/tokenizers/test_vocabulary.py` for the style.

---

## Pull requests

1. **Open or find an issue first** for anything non-trivial.
2. Branch from `main`, keep the change **focused on a single concern**.
3. Make sure the four gates pass and tests stay offline.
4. Update documentation touched by the change: the module README,
   `docs/design/` if a phase's design shifted, and `CHANGELOG.md` (under
   `[Unreleased]`).
5. Fill in the PR template — it mirrors this checklist.

Releases (version bumps, tags, PyPI publishing) are handled by the maintainer;
PRs should not bump versions.

## Reporting issues

Use the [issue templates](https://github.com/cattolatte/Polaris/issues/new/choose).
For bugs, include: what you expected, what happened, steps to reproduce, and your
environment (OS, Python, package version — `polaris info`).

---

## Code of Conduct

All contributors are expected to follow the project's
[**Code of Conduct**](CODE_OF_CONDUCT.md). We are committed to a welcoming and
respectful community.
