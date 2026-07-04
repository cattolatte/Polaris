# Phase 09 — Subword Tokenization (BPE)

**Status:** 🚧 In Progress

---

# Overview

The v0.8 benchmark showed both models capped at **~85.5%** by the **tokenization**:
whitespace splitting produces a large, sparse vocabulary with many
out-of-vocabulary (`<unk>`) tokens, so neither model can see past that ceiling.

v0.9 implements **Byte Pair Encoding (BPE)** from scratch — a subword tokenizer
that learns merges from the corpus, so rare and unseen words are split into known
subwords instead of collapsing to `<unk>`. It is the **second real tokenizer**,
dropping straight into the existing `Tokenizer` contract (v0.3): no model,
collation, training, or evaluation changes are needed. Re-running the benchmark
with BPE tests whether better *representation* — not a bigger model — breaks the
ceiling.

---

# Goals

At the end of Phase 9 Polaris can:

- **train** a BPE model (learn an ordered set of merges) from a corpus to a target
  vocabulary size, with reserved special tokens
- **tokenize / encode / decode** text with a `BPETokenizer` that satisfies the
  `Tokenizer` protocol and handles unknown symbols and padding
- select the tokenizer in the example (`whitespace` | `bpe`) and **re-benchmark**

**Proof of done:** BPE trains and tokenizes with far fewer `<unk>` than
whitespace; `BPETokenizer` satisfies the `Tokenizer` contract; the example trains
on BPE tokens; and the README benchmark is re-run with BPE (new, honest numbers).

---

# Non-Goals (deferred)

- **Byte-level BPE** (GPT-2 style over raw bytes), **WordPiece**, **SentencePiece**,
  **Unigram** — later tokenizers.
- Fast/parallel (Rust) training — this is the readable reference implementation.
- Pretrained merge tables.

---

# Abstraction decisions

- `BPETokenizer` is the **second concrete `Tokenizer`**. The protocol from v0.3
  already captures the shape (`tokenize` / `encode` / `decode` / `vocabulary`), so
  no new abstraction is introduced — the second implementation simply **validates**
  the protocol (ADR-0004).
- Training lives in a function/classmethod, separate from the tokenizer's runtime,
  mirroring how `build_vocabulary` is kept out of `Vocabulary`.

---

# Directory Structure

```
polaris/
│
└── tokenizers/
    ├── tokenizer.py     # protocol (v0.3)
    ├── whitespace.py    # v0.3
    └── bpe.py           # BPETokenizer + train (new)

tests/
└── unit/tokenizers/
    └── test_bpe.py
```

---

# Components (in build order)

1. **BPE training** — from word-frequency counts: represent each word as its
   characters plus an end-of-word marker, then iteratively merge the most frequent
   adjacent symbol pair (weighted by word frequency), recording each merge in
   order, until the vocabulary budget is reached. Special tokens (`<pad>`,
   `<unk>`) are reserved. Deterministic (ties broken consistently).
2. **`BPETokenizer`** — holds the learned merges and a `Vocabulary`. `tokenize`
   applies the merges greedily within each whitespace-separated word; `encode`
   maps subwords to ids (with `<unk>` fallback for unknown symbols); `decode`
   rejoins subwords and strips the end-of-word marker. Satisfies the `Tokenizer`
   protocol.
3. **Example integration & re-benchmark** — a `TOKENIZER` switch
   (`"whitespace"` | `"bpe"`); re-run the IMDB benchmark with BPE and update the
   README Results with the new numbers.

---

# Design Principles

- **From scratch, readable.** The classic word-level BPE (Sennrich et al.), by
  hand, on plain data structures — the point is to *understand* subwording.
- **Reuse the contract.** No changes to models, collation, training, or evaluation
  — BPE is just another `Tokenizer`.
- **Deterministic.** The same corpus and vocab size always learn the same merges.

---

# Testing Strategy

Offline, tiny corpora:

- Training learns expected merges and is deterministic across calls.
- A frequent word encodes to few subwords; an unseen word (of known characters)
  splits into subwords rather than a single `<unk>`.
- `encode` → `decode` reconstructs the text (up to whitespace normalization).
- `BPETokenizer` passes `isinstance(t, Tokenizer)`.
- The vocabulary respects the size budget and reserves the special tokens.

---

# Deliverables

- BPE training and `BPETokenizer` (from scratch)
- Example tokenizer switch and a re-run benchmark
- Updated README Results
- Complete offline tests, documentation, green CI

---

# What comes next

v0.10 (Deployment & CLI) exposes trained models through a matured `polaris` CLI
and serving.
