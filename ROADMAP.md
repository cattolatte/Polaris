# Polaris Roadmap

> Building a production-inspired, educational NLP engineering platform — one runnable milestone at a time.

---

## Overview

Polaris is developed incrementally with a strong emphasis on software engineering
quality, modular architecture, reproducibility, and maintainability.

Polaris is an **educational, reference-implementation-first** platform: the primary
product is the codebase itself — a from-scratch NLP system, clean enough to read as a
teaching text and engineered well enough to run a real task reproducibly. See
[ADR-0001](docs/adr/0001-project-identity.md).

Rather than rapidly adding features, each phase delivers a **runnable vertical
slice** — a working thread through the layers it touches — not scaffolding for a
future layer.

---

## Development Philosophy

Every phase of Polaris follows the same engineering workflow:

```text
Concrete Implementation
        ↓
Design Review
        ↓
Testing
        ↓
Documentation
        ↓
Code Review
        ↓
Release
        ↓
Extract Abstractions (when a second implementation proves the pattern)
```

No phase is considered complete until it has been documented, tested, and reviewed.

### Engineering Rules

These rules apply to every component (see also `CONTRIBUTING.md` and `docs/adr/`):

- **Concrete before abstract**: Build one real implementation first.
- **Evidence-driven abstraction**: Extract an abstraction only when **two or more**
  concrete implementations demand it ([ADR-0004](docs/adr/0004-abstraction-policy.md)).
- **Vertical slices over horizontal layers**: Every release leaves Polaris in a
  **runnable** state ([ADR-0002](docs/adr/0002-vertical-slice-architecture.md)). Never
  build a model you can't train, or a trainer with nothing to train.
- **Own the interface**: Third-party libraries (Hugging Face `datasets`, PyTorch) are
  implementation details, never public API.
- **Test behavior, not implementation**: Tests assert the public contract.
- **Offline unit tests**: No network access, no downloads. External backends are
  mocked.
- **Never introduce speculative infrastructure**: Configuration, logging, callbacks,
  registries, and the CLI are built when a real component requires them, not before.
- **Strict tooling**: Black, Ruff, MyPy (strict), and Pytest gate every merge.
- **Document major decisions**: significant architectural choices become ADRs.

---

## Current Status

| Item | Status |
|------|--------------------------|
| Current Version | `v0.11.0` |
| Development Stage | Deployment & CLI (v0.12, upcoming) |
| Overall Progress | 🚧 Active Development |

> **Roadmap revised 2026-07-03.** The future phases below were restructured from
> horizontal layers into vertical slices, and scope was reduced, following the
> architecture review captured in [ADR-0002](docs/adr/0002-vertical-slice-architecture.md)
> and [ADR-0003](docs/adr/0003-tensor-framework.md). Completed phases are unchanged.
> See **"What changed in this revision"** at the end.

---

## Long-Term Development Roadmap

## v0.0.1-alpha — Repository Initialization

Status: ✅ Completed

**Objectives**

- Initialize repository
- Define project architecture
- Establish package structure
- Create project documentation
- Prepare development environment

**Deliverables**

- Initial repository scaffold
- Documentation structure
- Community health files
- Project vision

---

## v0.1.0 — Foundation

Status: ✅ Completed

**Goal**

Build the minimal engineering foundation that every future Polaris module depends on.

**Major Components**

- Core framework (protocols, shared types)
- Exception hierarchy (`PolarisError` and subclasses)
- Registry system *(built here, now dormant — see [ADR-0005](docs/adr/0005-registry-dormant.md))*
- CI/CD pipeline
- Strict typing and quality tooling

**Deliberately Deferred**

- Configuration system → built when the Training Engine matures (v0.6)
- Logging → built when the Training Engine needs it (v0.6)
- CLI → introduced early and grown incrementally from v0.4 onward

**Outcome**

A stable, minimal engineering foundation.

---

## v0.2.0 — Data Foundations

Status: ✅ Completed

**Goal**

Establish the core data contract and prove it with one real dataset.

**Major Components**

- `TextSample` — immutable core data abstraction ✅
- `Dataset` protocol — read-only collection contract ✅
- `IMDBDataset` — first concrete dataset, backed by Hugging Face behind a
  Polaris-native interface ✅
- Offline test suite with mocked backend ✅

**Outcome**

A proven data contract and the template for every future dataset wrapper.

---

## v0.3.0 — Tokenization Foundations

Status: ✅ Completed

**Goal**

Build the tokenization contract and prove it with the simplest real tokenizer.

**Major Components (in order)**

1. Tokenizer interface ✅
2. Vocabulary ✅
3. Encoding abstraction ✅
4. Whitespace tokenizer (first concrete implementation) ✅

**Note**

The whitespace tokenizer is **sufficient to reach the first end-to-end slice (v0.4).**
Additional tokenizers (Character, BPE, WordPiece, SentencePiece, Unigram) are
**incremental v0.3.x work and are not blocking** — they are added opportunistically,
each as its own small slice, once the pipeline exists. Extracting further shared
tokenizer abstractions waits for a second real algorithm (per ADR-0004).

**Outcome**

A modular tokenization contract, proven by a real implementation, ready to feed the
first end-to-end pipeline.

---

## v0.4.0 — First End-to-End Slice ("Hello, IMDB")

Status: ✅ Completed — **the pivotal vertical slice**

**Goal**

Deliver the thinnest **runnable** thread from raw text to a measured result. After
this release, a user can train a real sentiment classifier on IMDB, end to end, in a
few lines. This is where PyTorch enters the project
([ADR-0003](docs/adr/0003-tensor-framework.md)).

**Major Components (each minimal, in dependency order)**

1. **Collation / batching** — the missing seam between tokenizers and models:
   padding, truncation, attention masks, and conversion to PyTorch tensors. Turns a
   sequence of `Encoding`s into a batch.
2. **One small from-scratch model** — the simplest thing that proves the pipeline
   (e.g. embedding → mean-pool → linear classification head). No transformer yet.
3. **Minimal training loop** — forward, loss, backward, optimizer step, over epochs.
   Explicit function arguments for hyperparameters — **no config system, no callbacks,
   no checkpointing, no distributed, no mixed precision.**
4. **Basic evaluation** — loss and accuracy on the test split.
5. **Reproducibility seed utility** — deterministic seeding (`utils`), the first
   concrete home for the reproducibility pillar.
6. **A thin `polaris` CLI entry** — e.g. `polaris info`; grown each subsequent phase.
7. **A runnable example / notebook** demonstrating the full slice.

**Explicitly deferred to later slices**

- Transformers, `Model`/`Trainer` abstractions, config system, checkpointing,
  callbacks, schedulers, experiment tracking.

**Outcome**

Polaris runs an NLP task end to end. Every later phase thickens one layer of this
proven pipeline.

---

## v0.5.0 — Transformer Encoder (from scratch)

Status: ✅ Completed

**Goal**

Thicken the model layer with the educational centerpiece: a transformer encoder
implemented **from scratch** on tensor primitives, reusing the v0.4 collation,
training, and evaluation harness unchanged (which proves the seams).

**Major Components**

- Scaled dot-product attention, multi-head attention (from scratch)
- Positional encoding, layer normalization, feed-forward block, residual connections
- A transformer-encoder text classifier
- Trained on IMDB and compared against the v0.4 baseline
- **Extract the `Model` abstraction now** — a second concrete model justifies it
  (ADR-0004)

**Outcome**

A readable, correct, from-scratch transformer, runnable end to end — the core teaching
artifact of Polaris.

---

## v0.6.0 — Training Engine Maturity

Status: ✅ Completed

**Goal**

Thicken the training layer into a real, reusable engine — driven by the fact that two
model trainings now exist.

**Major Components**

- Extract the `Trainer` abstraction (justified by ≥2 training uses)
- Checkpointing (save/restore)
- Callbacks / hooks (e.g. early stopping, metric logging)
- Learning-rate schedulers
- **Configuration system** — introduced *here*, where config-driven runs first add
  real value (small, typed, Pydantic-based)
- **Logging infrastructure** — introduced *here*, where it is first needed

**Deferred to post-1.0**

- Distributed / multi-GPU training, mixed precision (see Post-1.0)

**Outcome**

Production-inspired, single-device training infrastructure that stays runnable and
now reproducible from config.

---

## v0.7.0 — Evaluation Framework

Status: ✅ Completed

**Goal**

Grow the basic v0.4 metrics into a comprehensive, reusable evaluation framework.

**Major Components**

- Classification metrics (accuracy, precision/recall/F1, confusion matrix)
- An evaluation harness decoupled from training
- Benchmark reports
- **Essential visualization folded in here**: training curves and confusion-matrix
  plots (richer viz — attention/embeddings — is post-1.0)

**Outcome**

Reliable, reproducible measurement of model quality.

---

## v0.8.0 — Experiment Tracking & Reproducibility

Status: ✅ Completed

**Goal**

Make runs reproducible and comparable.

**Major Components**

- A Polaris-native tracking interface backed by **one** backend first (concrete before
  abstract — a second backend only if a real need appears)
- Run tracking, configuration snapshots, artifact management
- Reproducibility: seed capture, environment capture
- **Registry reconsidered here** — experiment tracking is the likely first real
  consumer of component discovery (ADR-0005); reactivate and validate against the need
  if it materializes

**Outcome**

Complete, reproducible experiment management.

---

## v0.9.0 — Subword Tokenization (BPE)

Status: ✅ Completed

**Goal**

Break the tokenization ceiling found in the v0.8 benchmark (both models capped at
~85.5%) by implementing **Byte Pair Encoding from scratch** — a subword tokenizer
that splits rare/unseen words into known subwords instead of `<unk>`.

**Major Components**

- BPE training (learn ordered merges from a corpus)
- `BPETokenizer` satisfying the existing `Tokenizer` contract (the second real
  tokenizer, validating the v0.3 protocol)
- A tokenizer switch in the example, and a **re-run benchmark** with BPE

**Outcome**

Better representation — not a bigger model — as the lever on accuracy, measured
honestly against the whitespace baseline.

---

## v0.10.0 — Pretrained Embeddings (GloVe)

Status: ✅ Complete

**Goal**

Test whether **pretrained GloVe word vectors** break the ~85–86% ceiling found in
the v0.9 benchmark by initializing the embedding layer with them instead of random
ones.

**Major Components**

- Load GloVe vectors and build an embedding matrix aligned to a `Vocabulary`
- `pretrained_embeddings` support on the models (optionally frozen)
- A GloVe-enabled example and a re-run benchmark

**Outcome**

Measured honestly: pretrained *word* embeddings do **not** break the ceiling on
IMDB (+0.001 for pooling; the transformer overfits harder, 0.849). With 25k
labeled reviews the model learns good embeddings from scratch anyway — pretrained
word vectors pay off when labeled data is scarce. This isolates the real lever:
self-supervised *contextual* pretraining (v0.11).

---

## v0.11.0 — Self-Supervised Pretraining (Masked Language Modeling)

Status: ✅ Complete

**Goal**

Break the ceiling the honest way, entirely **from scratch**: pretrain Polaris' own
transformer on **unlabeled** text with a masked-language-modeling objective, then
fine-tune it on the labeled sentiment task. This is the recipe that actually
transformed NLP (BERT) — implemented as our own code, on our own data, producing
our own weights. Nothing is downloaded; "pretraining" here is a *training method*,
not a borrowed model.

**Major Components**

- A `<mask>` special token and MLM collation: randomly mask ~15% of tokens
  (BERT-style 80/10/10) and produce prediction targets for the masked positions
- Extract the shared transformer **trunk** (embedding + positional + encoder
  blocks → per-token hidden states) now that a *second* consumer (the MLM head)
  proves the shape — the classifier and the MLM model both reuse it (ADR-0004)
- A masked-language-model head + model, and a pretraining loop (cross-entropy on
  masked positions only)
- Transfer the pretrained trunk into the classifier and fine-tune
- Pretrain on the **unlabeled IMDB split** (50k reviews), fine-tune on the 25k
  labeled set, and benchmark against the from-scratch baseline

**Outcome**

Contextual pretraining demonstrated on Polaris' own from-scratch stack, with the
whole pretrain → fine-tune thread runnable end to end. A controlled ablation (same
vocabulary and 4-layer architecture) shows pretraining works as designed — a large
warm start (epoch-1 validation 0.810 vs 0.736), faster convergence, higher best
validation (0.864 vs 0.851) — but converges to the same ~86% test ceiling (0.853 vs
0.852): 25k labels already suffice, and the small in-domain corpus injects little
new knowledge. This closes an honest four-lever investigation (transformer, BPE,
GloVe, MLM pretraining), all landing at ~85-86% — the ceiling is the task and
data/compute regime, not any one component. Breaking it further would require
large-external-corpus pretraining beyond the from-scratch-on-a-laptop scope.

---

## v0.12.0 — Deployment & CLI

Status: ⏳ Planned

**Goal**

Make trained models usable outside a script, via production-inspired workflows.

**Major Components**

- Inference runtime (`inference/`) — run predictions with a trained model
- The `polaris` CLI matured into the primary UX (it has been growing since v0.4):
  train, evaluate, predict
- Packaging: FastAPI/REST serving and Docker
- ONNX export *(optional / best-effort — kept out of the critical path)*

**Outcome**

A trained Polaris model can be served and invoked as a real product.

---

## v1.0.0 — Polaris Stable

Status: 🔒 Future

**Goal**

Release the first stable version: one cohesive, from-scratch NLP system that reads as
a teaching text and runs a real task reproducibly, end to end.

**Objectives**

- Stable public APIs
- Comprehensive documentation and tutorials
- Complete testing (with a CI matrix and coverage gate)
- Modular architecture, reproducible workflows
- Public release

**Outcome**

First production-ready public release.

---

## Post-1.0 / Future Vision

Deliberately **out of scope** before v1.0 to protect quality and focus. Each would
arrive as its own vertical slice:

- **More architectures**: decoder / causal LM, seq2seq, text generation.
- **More tokenizers**: the full BPE / WordPiece / SentencePiece / Unigram family
  (beyond the incremental v0.3.x additions), plus tokenizer training and benchmarking.
- **Scale**: distributed / multi-GPU training, mixed precision.
- **Richer visualization**: attention maps, embedding projections, error analysis.
- **Additional tracking backends** (e.g. MLflow *and* TensorBoard).
- **Plugin ecosystem**, interactive dashboards, benchmark leaderboards, model/dataset
  hub integrations, advanced MLOps.

These are intentionally excluded from the initial v1.0 release.

---

## Cross-Cutting Quality Goals

Ongoing, not tied to a single phase:

- **CI hardening**: expand from single-version to a Python/OS matrix; enforce a
  coverage threshold.
- **ADRs**: every major architectural decision is recorded in `docs/adr/`.
- **Just-in-time design docs**: each phase's `docs/design/phase_NN_*.md` is written
  immediately before implementation, not in advance.

---

## What changed in this revision (2026-07-03)

For transparency, the future roadmap was restructured and **scope was reduced**:

- **Horizontal layers → vertical slices.** The old "Model Zoo → Training → Evaluation"
  layering is replaced by an early end-to-end slice (v0.4) that is then thickened.
- **Model Zoo trimmed.** Six planned architectures become one baseline classifier
  (v0.4) + one from-scratch transformer encoder (v0.5). Decoder / seq2seq / generation
  move to Post-1.0. **"Foundation model interfaces" removed** pre-1.0.
- **Metrics and config repositioned.** Basic metrics arrive with the first slice
  (v0.4); the config system arrives with the mature trainer (v0.6), not after a
  separate training phase.
- **Collation layer added** — previously missing between tokenizers and models.
- **Experiment tracking reduced** to a single backend first.
- **Distributed training & mixed precision deferred** to Post-1.0.
- **Standalone Visualization phase removed** — essentials folded into Evaluation
  (v0.7); the rest is Post-1.0.
- **CLI pulled earlier** — a thin CLI from v0.4, grown per phase, instead of a v0.9
  big-bang.
- **Tensor framework decided**: PyTorch, with model internals from scratch (ADR-0003).

---

## Guiding Principles

Every feature added to Polaris should satisfy the following principles:

- **Simplicity**: Prefer simple, clear solutions over unnecessary complexity.
- **Modularity**: Design components that are loosely coupled and independently
  maintainable.
- **Readability**: Write code that is easy for others to understand — the reader is a
  learner.
- **Reproducibility**: Ensure that experiments and results are repeatable.
- **Documentation**: Treat documentation as a core part of the implementation.
- **Testing**: Validate functionality and prevent regressions through a robust test
  suite.
- **Quality**: Prioritize engineering quality over development speed.
- **Evidence-driven abstraction**: Abstractions are extracted from working code, never
  designed in advance of it.

---

## Project Status Legend

| Symbol | Meaning |
|---------|---------|
| ✅ | Completed |
| 🚧 | In Progress |
| ⏳ | Planned |
| 🔒 | Future |

---

> Polaris is a long-term engineering project focused on understanding, building, and
> maintaining modern NLP systems through production-inspired software engineering
> practices.
