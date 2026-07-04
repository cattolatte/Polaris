# Phase 05 — Transformer Encoder (from scratch)

**Status:** ✅ Completed

> The educational centerpiece. See [ADR-0003](../adr/0003-tensor-framework.md)
> (model internals from scratch) and [ADR-0004](../adr/0004-abstraction-policy.md)
> (extract the `Model` abstraction now that a second model exists).

---

# Overview

v0.4 proved the pipeline with a deliberately trivial model (mean-pooled
embeddings). v0.5 replaces it with the real thing: a **transformer encoder,
implemented from scratch** on PyTorch tensor primitives.

Crucially, this phase reuses v0.4's collation, training loop, and evaluation
**unchanged** — swapping only the model. That reuse is the whole point of a
vertical slice: if the transformer drops into the existing harness and trains,
the seams were right.

This is the most important teaching artifact in Polaris: a reader can follow
attention, multi-head attention, and a transformer block line by line.

---

# Goals

At the end of Phase 5 Polaris can:

- compute scaled dot-product attention, from scratch, with masking
- compose it into multi-head self-attention
- build a transformer encoder block (attention + feed-forward, residual + norm)
- stack blocks into a transformer-encoder text classifier
- train it on IMDB with the existing loop and beat the v0.4 baseline
- express both models through a shared `Model` abstraction (extracted now that a
  second implementation exists)

---

# Non-Goals

- Decoder / causal attention, seq2seq, generation — Post-1.0
- Pretraining objectives (MLM, etc.) — later
- KV-caching, flash-attention, or any performance optimization
- `nn.Transformer` / `nn.MultiheadAttention` — we implement these ourselves
  (that is the point); the framework provides only tensor ops, autograd, and
  optimizers
- Configuration system, checkpointing, schedulers — v0.6

---

# Dependency direction

Unchanged from v0.4:

```
data → tokenizers → collation → models → training → evaluation
```

Only the `models` layer grows. Collation, training, and evaluation are reused
as-is.

---

# Directory Structure

```
polaris/
│
├── models/
│   ├── __init__.py
│   ├── model.py                  # Model protocol (extracted; Batch -> logits)
│   ├── pooling_classifier.py     # existing v0.4 baseline
│   ├── attention.py              # scaled dot-product + multi-head attention
│   ├── transformer.py            # positional encoding, block, layer norm, FFN
│   └── transformer_classifier.py # TransformerEncoderClassifier
│
examples/
└── train_imdb_transformer.py     # or a flag on the existing example

tests/
└── unit/
    └── models/
        ├── test_attention.py
        ├── test_transformer.py
        └── test_transformer_classifier.py
```

(Exact file split may be adjusted during implementation; the point is small,
readable units.)

---

# Components (in build order)

Each is a self-contained sub-slice: implement from scratch, test invariants,
then compose.

1. **Scaled dot-product attention** — `softmax(QKᵀ / √d) V`, with support for a
   key-padding mask (padding positions get ~-inf before softmax). Tested for
   shape, that attention weights sum to 1, and that masked positions are ignored.
2. **Multi-head attention** — split into heads, attend per head, concatenate,
   project. Tested for shape and that it reduces to single-head behaviour when
   `num_heads=1`.
3. **Transformer block** — self-attention + position-wise feed-forward, each
   wrapped in a residual connection and layer normalization (layer norm
   implemented from scratch). Tested for shape preservation and that padding
   does not leak into non-padded outputs.
4. **Transformer encoder classifier** — token embeddings + positional encoding →
   N blocks → masked pooling → linear head. Consumes a `Batch`, returns logits;
   drop-in compatible with the v0.4 training loop.
5. **`Model` abstraction — deferred (decision).** Two concrete models now exist,
   but no *consumer* needs a Polaris-specific model protocol: the training loop
   and evaluation already treat models polymorphically via `nn.Module` (the real
   shared contract is "an `nn.Module` whose `forward` takes a `Batch` and returns
   logits"). Per ADR-0004 and the ADR-0005 lesson, extracting a protocol with no
   consumer would be speculative — a dead abstraction. We therefore **defer** it
   until something actually needs it (e.g. a model registry or config-driven
   model selection in v0.6+). The N≥2 rule is necessary, not sufficient: there
   must also be a consumer.
6. **Example** — train the transformer on IMDB, reusing the collation, training,
   and evaluation from v0.4, and report the same metrics for comparison against
   the baseline.

---

# Design Principles

- **From scratch where it teaches.** Attention, multi-head, the block, and layer
  norm are hand-written on tensor primitives. Framework used only for autograd,
  `optim`, embeddings, and linear layers.
- **Reuse proves the seams.** Collation, training, and evaluation are not
  modified. If they need changing, that is a signal to examine the abstraction.
- **Abstractions need a consumer, not just two implementations.** We *defer* a
  `Model` protocol: two models exist, but nothing consumes them through a Polaris
  interface (see component 5).
- **Readable over fast.** No performance tricks; the naive, correct formulation.

---

# Testing Strategy

Offline, on tiny fixtures. Assert shapes and invariants, not exact floats:

- Attention weights form a valid distribution (sum to 1 along keys).
- A causal/padding mask actually zeroes the masked contributions.
- Multi-head output shape equals input shape; `num_heads=1` matches single-head.
- A block preserves `(batch, seq, dim)` and is permutation-consistent where
  expected.
- The classifier returns `(batch, num_classes)` and trains (loss decreases) on a
  tiny synthetic dataset via the existing `train` loop.

---

# Deliverables

- From-scratch attention, multi-head attention, transformer block, layer norm
- `TransformerEncoderClassifier`
- A documented decision to defer the `Model` abstraction (no consumer yet)
- A transformer training example reusing the v0.4 harness
- Complete offline tests, documentation, green CI

**Proof of done:** the transformer trains on IMDB through the existing loop and
reports metrics that beat the v0.4 mean-pooling baseline.
