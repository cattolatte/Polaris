# Phase 13 — v1.0.0 Stabilization

Status: 🚧 In Progress · Target: **v1.0.0** (first stable release)

> Written just before the work, per the docs convention.

## Why this phase

Every feature slice of the roadmap is done: data, tokenization, collation, models
(from-scratch transformer), training, evaluation, experiment tracking, pretrained
embeddings, self-supervised pretraining, and deployment. v1.0 adds **no features**.
It is the graduation pass: freeze the public API, finish the documentation, harden
the test setup, and make the whole thing read and run as the cohesive teaching text
Polaris set out to be.

## Non-goals

- **No new features.** The interactive console and PyPI publishing are explicitly
  *out* — they are their own later versions (v1.x).
- **No re-architecting.** Stabilization polishes what exists; it does not move it.

## Work

### 1. API freeze

Polaris' public API is its **submodule surface**: each module exports an intentional
`__all__` (e.g. `from polaris.inference import Predictor, load_bundle`). This is a
deliberate design choice over a large top-level `polaris` namespace: importing
`polaris` must stay cheap and must **not** require the optional `torch` extra (so
`polaris info` and `polaris.__version__` work in a bare install). The top-level
package therefore intentionally exports only `__version__`; heavy, optional-dep
modules are imported from their subpackages.

The freeze is a review, not a rewrite: confirm every public `__all__` is intentional
and documented; nothing is renamed. From v1.0, these submodule surfaces are stable
under semver.

### 2. Documentation pass

- Fix module READMEs that still read "Planned / Not implemented" although the module
  is fully built: `core`, `models`, `evaluation`, `experiments`, `utils`.
- Add the missing module READMEs: `collation`, `embeddings`.
- `visualization` stays a *genuine* planned stub (that module has no code yet); its
  "Planned" README is correct and remains.
- Add an **Installation** and a **Quickstart** ("train and serve your first model")
  to the top-level README.

### 3. Test hardening

- Add a **CI matrix**: Python 3.12 and 3.13, on Ubuntu and macOS (`fail-fast:
  false`), so cross-version/cross-OS breakage is caught.
- Keep the coverage gate: CI already runs `pytest --cov=polaris`, which honors
  `fail_under = 90` from `pyproject.toml`.

### 4. Consistency sweep

- Update the repository guidance to reflect the finished stack.
- Mark v1.0.0 complete in ROADMAP; update the header.
- CHANGELOG: promote to `v1.0.0` with a summary of the journey.

### 5. Release

Version → `1.0.0`, tag, and GitHub Release (the full ceremony). PyPI publishing is
deferred to v1.0.1 by design.

## Definition of done

All four gates green on the matrix; no implemented module advertises itself as
"planned"; a newcomer can read the READMEs and run the quickstart end to end; the
public API is documented and frozen under semver.
