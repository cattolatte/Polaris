# Changelog

All notable changes to Polaris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where appropriate. During the `0.x.y` development phase, the API should be considered unstable, and breaking changes are expected.

---

## [Unreleased]

Work toward **v0.3.0 — Tokenization Foundations**. See `ROADMAP.md`.

### Added

- Tokenization foundations: the `Tokenizer` protocol, `Vocabulary`, `Encoding`, and the `WhitespaceTokenizer` reference implementation.
- Public API re-exports so components import from their package roots: `polaris.tokenizers` (`Tokenizer`, `Vocabulary`, `Encoding`, `WhitespaceTokenizer`), `polaris.data` (`Dataset`, `TextSample`), and `polaris.data.datasets` (`IMDBDataset`).

### Changed

- Unified the project version to `0.3.0.dev0` across `version.py`, `pyproject.toml`, `uv.lock`, and the README version badge.
- Updated the Phase 01 and Phase 03 design docs to match the shipped code and the roadmap's deferral of configuration, logging, and the CLI.
- Replaced the placeholder `polaris/tokenizers/README.md` with real module documentation.

### Fixed

- Root-anchored the ML-artifact patterns in `.gitignore` (`/datasets/`, `/artifacts/`, `/checkpoints/`, `/runs/`, `/mlruns/`) so they no longer hide the `polaris/data/datasets/` source package or its tests. The IMDB dataset implementation is now tracked in version control.
- Renamed the misspelled `polaris/training/__innit__.py` to `__init__.py`.
- Brought the tokenizer and IMDB modules into Black and Ruff compliance (the IMDB files had never been linted because they were being ignored).

---

## [v0.2.0] - 2026-07-03

Data Foundations. Established the core data contract and proved it with one real dataset.

### Added

- `TextSample`: the immutable, strongly typed representation of a single labeled text example, with read-only metadata.
- `Dataset` protocol: the read-only collection contract (`name`, `__len__`, `__getitem__`, `__iter__`) implemented by every dataset.
- `IMDBDataset`: the first concrete dataset, backed by the Hugging Face `datasets` library behind a Polaris-native interface.
- Offline unit test suite for the data layer, using an in-memory fake backend (no network access).
- Optional `datasets` dependency extra.

### Notes

- Dataset registry integration, transforms, splitting utilities, and caching were deliberately deferred until multiple datasets exist (see `ROADMAP.md`).

---

## [v0.1.0] - 2026-07-02

Foundation. The minimal engineering foundation every future module depends on.

### Added

- Core framework: the `Component` protocol and shared types (`ComponentMetadata`, `ComponentType`, and framework-wide type aliases).
- `PolarisError` exception hierarchy (`DatasetError`, `InvalidSplitError`, `MissingDependencyError`).
- Component `Registry` for explicit registration and lookup of framework components, with no global mutable state.
- Continuous Integration (GitHub Actions) enforcing Black, Ruff, MyPy (strict), and Pytest on Python 3.12.
- Strict typing and quality-tooling configuration.

### Notes

- Configuration, logging, and the CLI were deliberately deferred to later phases where a real component first requires them (see `ROADMAP.md`).

---

## [v0.0.1-alpha] - 2026-07-01

### Added

- Initialized the Polaris repository.
- Established the high-level project architecture and long-term vision.
- Created the initial package structure for future modules.
- Added foundational project documentation (`README.md`, `ROADMAP.md`).
- Added community health files (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`).

### Notes

This release marks the formal beginning of the Polaris project. No production-ready functionality has been implemented. Development will proceed incrementally according to the **Project Roadmap**, with each milestone focusing on software quality, maintainability, and engineering best practices.