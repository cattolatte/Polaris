# Changelog

All notable changes to Polaris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where appropriate. During the `0.x.y` development phase, the API should be considered unstable, and breaking changes are expected.

---

## [v0.4.0] - 2026-07-04

First End-to-End Slice. Polaris now runs a real NLP task end to end: from raw
IMDB reviews to a trained sentiment classifier and a measured accuracy, using a
from-scratch model on PyTorch with Apple Silicon (MPS) / CUDA / CPU support.

### Added

- Collation layer (`polaris.collation`): `Batch` and `collate`, turning tokenizer output into padded, model-ready PyTorch tensors with an attention mask.
- `MeanPoolingClassifier` (`polaris.models`): the first from-scratch model — embedding, mask-aware mean pooling, and a linear classification head.
- A minimal training loop (`polaris.training.train`) and basic metrics (`polaris.evaluation.accuracy` / `evaluate`).
- Deterministic seeding (`polaris.utils.set_seed`).
- Device selection (`polaris.utils.resolve_device`) preferring Apple Silicon MPS, then CUDA, then CPU, plus `Batch.to`. Training and evaluation move batches to the model's device automatically, and the example uses the best available device.
- A thin `polaris` command-line interface with an `info` command.
- `examples/train_imdb_sentiment.py`: a complete, runnable end-to-end sentiment-training script on IMDB.
- An offline end-to-end integration test that exercises the full pipeline without downloading a dataset.
- PyTorch as an optional `torch` extra, resolved to CPU-only wheels on Linux.

### Changed

- CI installs the `torch` extra so the collation/model/training tests run.

### Fixed

- `IMDBDataset` loads the canonical `stanfordnlp/imdb` Hugging Face repo id; the bare `imdb` id is rejected by newer `datasets`/`huggingface_hub` versions.

---

## [v0.3.0] - 2026-07-03

Tokenization Foundations. The tokenizer contract, proven with a real
implementation, plus the vocabulary tooling required to train on a real dataset.
Also adopts the vertical-slice architecture and records the project's major
decisions as ADRs.

### Added

- Tokenization foundations: the `Tokenizer` protocol, `Vocabulary`, `Encoding`, and the `WhitespaceTokenizer` reference implementation.
- Special tokens on `Vocabulary`: optional `unk_token` and `pad_token` (with derived `unk_id` / `pad_id` and a `special_tokens` property), plus an unknown-aware `get_id`. `WhitespaceTokenizer.encode` now maps out-of-vocabulary tokens to the unk id when one is configured, and still raises otherwise.
- `build_vocabulary` (`polaris.tokenizers`), which builds a `Vocabulary` from a tokenized corpus with frequency ordering, deterministic tie-breaking, `min_frequency`, `max_size`, and reserved special tokens — the prerequisites for training on a real dataset.
- Public API re-exports so components import from their package roots: `polaris.tokenizers` (`Tokenizer`, `Vocabulary`, `Encoding`, `WhitespaceTokenizer`, `build_vocabulary`), `polaris.data` (`Dataset`, `TextSample`), and `polaris.data.datasets` (`IMDBDataset`).
- Architecture Decision Records under `docs/adr/`, capturing the project identity, vertical-slice architecture, the PyTorch tensor-framework decision (with model internals from scratch), the evidence-driven abstraction policy, and the dormant-Registry decision.

### Changed

- Unified the project version to `0.3.0.dev0` across `version.py`, `pyproject.toml`, `uv.lock`, and the README version badge.
- Restructured the future roadmap from horizontal layers into vertical slices and reduced scope, introducing an early end-to-end pipeline milestone (see the "What changed in this revision" section of `ROADMAP.md`).
- Documented the `polaris/registry/` module as dormant per ADR-0005 and replaced its placeholder README with an accurate one.
- Aligned the README with the new direction: marked the registry dormant, listed `docs/adr/`, and updated the engineering-philosophy pillars (vertical slices, evidence-driven abstraction).
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