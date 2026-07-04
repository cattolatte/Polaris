# Changelog

All notable changes to Polaris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where appropriate. During the `0.x.y` development phase, the API should be considered unstable, and breaking changes are expected.

---

## [v0.8.0] - 2026-07-04

Experiment Tracking & Reproducibility. Config snapshots, environment capture, and
a local run tracker make each run a reproducible artifact — and the README now
carries a reproducible IMDB benchmark.

### Added

- Config snapshots: `TrainingConfig.to_file` / `TrainingConfig.from_file` (JSON).
- `capture_environment` (`polaris.experiments`): records the Polaris, Python, PyTorch, and platform versions for reproducibility.
- A local experiment tracker (`polaris.experiments`): `record_run` writes a run's config, metric history, classification report, and environment to a directory; `load_run` reads them back into a `RunData`.

### Changed

- The IMDB example records each run (config, metrics, report, environment, seed) to `runs/imdb_<model>/` for reproducibility.
- Added a reproducible **Benchmarks** section to the README (IMDB: transformer and baseline tie at ~85.5% test accuracy, capped by whitespace tokenization).

---

## [v0.7.0] - 2026-07-04

Evaluation Framework. The metric primitives become a structured, reusable
`ClassificationReport` (per-class and averaged precision/recall/F1, support,
accuracy, confusion matrix) with a readable rendering, plus an `evaluate_model`
harness — the rigorous measurement layer a benchmark rests on.

### Added

- `ClassificationReport` (`polaris.evaluation`): a structured report with per-class and macro/weighted-averaged precision, recall, and F1, plus support, accuracy, and the confusion matrix, with a readable `to_text()` rendering.
- `classification_report(...)` builds a report from logits and labels; `evaluate_model(...)` is a harness that runs a model over batches and returns one — reusing the existing metric primitives.

### Changed

- The IMDB example prints a full `ClassificationReport` (per-class and averaged precision/recall/F1, support, accuracy, confusion matrix) instead of hand-assembling the metrics.

---

## [v0.6.0] - 2026-07-04

Training Engine Maturity. A real `Trainer` — warmup scheduling, validation, early
stopping, best-model checkpointing, and config-driven, logged runs — that fixes
the transformer's underfitting. Configuration and logging arrive here, where the
engine first needs them.

### Added

- `WarmupSchedule` (`polaris.training`): a from-scratch learning-rate schedule with linear warmup then cosine/linear decay — the ingredient the transformer needs to train well.
- Checkpointing (`polaris.training.save_checkpoint` / `load_checkpoint`): save and restore model + optimizer state with metadata.
- `get_logger` (`polaris.utils`): a standard-library logger for the training engine.
- `TrainingConfig` (`polaris.training`): a typed, validated Pydantic configuration for a run — Polaris' configuration system, introduced where first needed.
- `Trainer` (`polaris.training`): a training engine composing warmup scheduling, a validation loop, early stopping, best-model checkpointing, and logging (with `TrainingResult` / `EpochRecord`).

### Changed

- The IMDB example trains through the `Trainer` — a validation split, warmup scheduling, early stopping, and best-model selection. Transformer defaults raised (10 epochs, learning rate 1e-3) now that warmup makes a higher rate safe.

---

## [v0.5.0] - 2026-07-04

Transformer Encoder (from scratch). Polaris' educational centerpiece: a
transformer implemented on tensor primitives, trained through the unchanged v0.4
pipeline.

### Added

- From-scratch attention (`polaris.models.attention`): `scaled_dot_product_attention` and `MultiHeadSelfAttention`.
- From-scratch transformer building blocks (`polaris.models.transformer`): `LayerNorm`, `SinusoidalPositionalEncoding`, and a pre-norm `TransformerEncoderBlock`.
- `TransformerEncoderClassifier` (`polaris.models`) — a transformer text classifier that reuses the v0.4 collation, training, and evaluation unchanged.

### Changed

- The IMDB example trains either the transformer (default) or the pooling baseline via a `MODEL` switch, reusing the same data, collation, training, and evaluation code.

### Notes

- The `Model` abstraction is deliberately **deferred**: two models now exist, but no consumer needs a Polaris-specific protocol (they share the `nn.Module` contract). Extracting one now would be a speculative dead abstraction — see the Phase 05 design doc.

---

## [v0.4.1] - 2026-07-04

Richer evaluation and a regularized baseline, on top of the v0.4.0 slice.

### Added

- Classification metrics in `polaris.evaluation`: `predict` (gather logits and labels across batches), `confusion_matrix`, and per-class `precision_recall_f1`.

### Changed

- The IMDB example prints a per-class metrics table and a confusion matrix, and applies light regularization (a `min_frequency` cutoff when building the vocabulary and optimizer weight decay) to curb overfitting.

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