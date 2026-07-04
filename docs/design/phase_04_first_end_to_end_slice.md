# Phase 04 — First End-to-End Slice

**Status:** ✅ Completed

> The pivotal vertical slice. See [ADR-0002](../adr/0002-vertical-slice-architecture.md)
> (vertical slices) and [ADR-0003](../adr/0003-tensor-framework.md) (PyTorch,
> model internals from scratch).

---

# Overview

Until now Polaris can load a dataset (`IMDBDataset`), represent text
(`TextSample`), and turn text into ids (`WhitespaceTokenizer`, `Vocabulary`,
`build_vocabulary`). None of it *does* anything yet — there is no model and no
training.

Phase 4 delivers the **thinnest runnable thread** through every layer: from raw
IMDB reviews to a trained sentiment classifier and a measured accuracy. It is
the first release where Polaris runs an end-to-end NLP task.

This is deliberately a *thin* slice. Each layer is the simplest thing that makes
the pipeline work; later phases thicken individual layers (a real transformer in
v0.5, a mature trainer in v0.6, a full evaluation framework in v0.7).

PyTorch enters the project here as an optional dependency.

---

# Goals

At the end of Phase 4 Polaris can:

- turn a batch of `Encoding`s into padded PyTorch tensors (collation)
- run a small model, implemented on tensor primitives, over that batch
- train the model with a minimal training loop
- report loss and accuracy
- seed all randomness for reproducible runs
- do all of the above from a single runnable example script

Concretely: **train a sentiment classifier on IMDB and print its test accuracy.**

---

# Non-Goals

Phase 4 intentionally does NOT include (each has a later home):

- Transformers / attention — v0.5
- A `Model` or `Trainer` abstraction — extracted once a second implementation
  exists (v0.5 / v0.6), per [ADR-0004](../adr/0004-abstraction-policy.md)
- Configuration system, logging — v0.6
- Checkpointing, callbacks, learning-rate schedulers — v0.6
- Distributed / multi-GPU training, mixed precision — post-1.0
- Experiment tracking — v0.8
- A comprehensive metrics suite (precision/recall/F1, confusion matrix) — v0.7
- GPU-specific code paths, performance tuning

The model is chosen to be *boring*: an embedding, a masked mean-pool, and a
linear head. It exists to prove the pipeline, not to be accurate.

---

# Dependency direction

```
data  →  tokenizers  →  collation  →  models  →  training  →  evaluation
                                                     ↑
                                        utils (seeding) — cross-cutting
```

New layers follow the existing one-way rule: each depends only on layers to its
left plus `core`/`errors`. PyTorch is an implementation detail imported where
needed; the `data` and `tokenizers` layers stay PyTorch-free and offline.

---

# Directory Structure

```
polaris/
│
├── collation/
│   ├── __init__.py
│   ├── batch.py          # Batch: input_ids, attention_mask, labels (tensors)
│   └── collator.py       # collate(...): list of (Encoding, label) -> Batch
│
├── models/
│   ├── __init__.py
│   └── pooling_classifier.py   # MeanPoolingClassifier (nn.Module)
│
├── training/
│   ├── __init__.py
│   └── loop.py           # train(...): minimal training loop (function, not class)
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py        # accuracy(...), evaluate(...)
│
├── utils/
│   ├── __init__.py
│   └── seed.py           # set_seed(...)
│
└── cli.py                # thin Typer app: `polaris info`

examples/
└── train_imdb_sentiment.py   # the end-to-end script

tests/
└── unit/
    ├── collation/test_batch.py, test_collator.py
    ├── models/test_pooling_classifier.py
    ├── training/test_loop.py
    ├── evaluation/test_metrics.py
    └── utils/test_seed.py
```

---

# Components (in build order)

Each component is a self-contained sub-slice: implement, test, then move on.
The pipeline only has to run at the end of the phase, but every step is merged
green.

## 1. Collation (`polaris/collation/`)

Turns variable-length tokenizer output into a rectangular tensor batch — the
missing seam between tokenizers and models.

- **`Batch`** — a value object holding three aligned tensors: `input_ids`
  `(batch, seq_len)`, `attention_mask` `(batch, seq_len)` (1 for real tokens,
  0 for padding), and `labels` `(batch,)`. Immutable in spirit; because PyTorch
  tensors do not support value equality, this is a `slots` class without
  dataclass `__eq__` (or `eq=False`), not a hashable value object.
- **`collate(samples, *, pad_id, max_length=None)`** — takes a sequence of
  `(Encoding, label)` pairs, pads each sequence to the longest in the batch
  (or truncates to `max_length` if given), builds the attention mask, and
  stacks everything into tensors using `pad_id` from the vocabulary.

## 2. Model (`polaris/models/`)

- **`MeanPoolingClassifier(nn.Module)`** — `nn.Embedding` → mask-aware mean pool
  over the sequence → `nn.Linear` to `num_classes`. `nn.Embedding` and
  `nn.Linear` are undifferentiated primitives (plumbing), so their use is
  consistent with ADR-0003; the interesting, hand-written part is the
  masked pooling. No transformer, no attention yet.

## 3. Training loop (`polaris/training/`)

- **`train(model, batches, *, optimizer, epochs, ...)`** — a plain function:
  for each epoch, for each batch, forward → cross-entropy loss → backward →
  optimizer step. Returns per-epoch losses. **No `Trainer` class**,
  no checkpointing, no callbacks. Hyperparameters are explicit arguments.

## 4. Evaluation (`polaris/evaluation/`)

- **`accuracy(logits, labels)`** and a small **`evaluate(model, batches)`** that
  returns loss and accuracy. This is the seed of the v0.7 evaluation framework,
  kept minimal here.

## 5. Reproducibility (`polaris/utils/`)

- **`set_seed(seed)`** — seeds Python `random`, PyTorch (and NumPy if used), and
  sets deterministic flags. The first concrete home for the reproducibility
  pillar.

## 6. CLI (`polaris/cli.py`)

- A thin Typer app exposing `polaris info` (name, version, a one-line
  description). The `polaris` entry point already exists in `pyproject.toml`.
  The CLI grows one command per future phase; it is intentionally tiny here.

## 7. Example (`examples/train_imdb_sentiment.py`)

- The runnable proof: load IMDB, build a vocabulary from the training split
  (with `<pad>`/`<unk>`), tokenize + collate into batches, train the classifier,
  and print test accuracy. Small defaults so it runs on CPU in reasonable time
  (e.g. a capped number of samples / vocab size).

---

# Dependencies

- Add **`torch`** as an optional extra in `pyproject.toml`
  (e.g. `[project.optional-dependencies] torch`). It is imported inside the
  collation / models / training modules, never at the `data` or `tokenizers`
  layer.
- CI installs the `torch` extra so the model/training tests run. All tests stay
  **offline** — PyTorch is a local dependency, not a network call, and tests use
  tiny synthetic tensors, never downloaded weights.

---

# Design Principles

- **From scratch where it teaches; framework for plumbing.** Hand-write the
  masked pooling and the training loop; use `nn.Embedding`, `nn.Linear`,
  autograd, and `optim` as substrate (ADR-0003).
- **Thin slice.** Every layer is the minimum that makes the thread run.
- **No premature abstraction.** One model, one loop, no interfaces yet (ADR-0004).
- **Runnable at the end.** The phase is not done until the example script trains
  and reports accuracy.

---

# Testing Strategy

Offline, behaviour-focused, on tiny fixtures. Assert shapes, invariants, and
*learning behaviour* — never exact float values.

- **Collation**: padding length, `pad_id` placement, attention-mask correctness,
  truncation at `max_length`, output shapes, single-item and empty batches.
- **Model**: forward runs, output shape `(batch, num_classes)`, masked pooling
  ignores padded positions (padding does not change the pooled result).
- **Training**: loss decreases across epochs on a tiny synthetic, separable
  dataset; a fixed seed makes a run reproducible.
- **Metrics**: `accuracy` on known predictions.
- **Seed**: `set_seed` makes two subsequent random draws identical.
- **CLI**: `info` runs and reports the version.

---

# Deliverables

- Collation (`Batch`, `collate`)
- `MeanPoolingClassifier`
- Minimal `train` loop
- Basic `accuracy` / `evaluate`
- `set_seed`
- Thin `polaris info` CLI
- `examples/train_imdb_sentiment.py`
- Complete unit tests, documentation, green CI (with the `torch` extra)

**Proof of done:** running `examples/train_imdb_sentiment.py` trains the
classifier on IMDB and prints a test accuracy.

---

# What comes next

v0.5 replaces the pooling model with a **from-scratch transformer encoder**,
reusing this phase's collation, training loop, and metrics unchanged — which is
exactly how a vertical slice proves its seams.
