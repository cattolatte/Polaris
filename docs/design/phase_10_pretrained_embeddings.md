# Phase 10 — Pretrained Embeddings (GloVe)

**Status:** 🚧 In Progress

---

# Overview

The v0.9 benchmark established that the ~85–86% ceiling on IMDB is the **model
class** — simple, from-scratch models with **no pretraining** — not the
tokenization. The single lever that breaks that ceiling is **pretrained
representations**: starting the embedding layer with vectors that already encode
word meaning, rather than random noise.

v0.10 adds the ability to initialize the embedding layer from **pretrained GloVe
word vectors**. The model no longer has to learn what "great" and "terrible" mean
from 25k reviews — it starts knowing. This is the reason modern NLP works,
demonstrated on our own from-scratch stack.

We *consume* pretrained vectors (a standard, honest technique); we do not train
them. The vectors are a small, well-defined input, not a new model.

---

# Goals

At the end of Phase 10 Polaris can:

- **load** GloVe vectors from a file
- **build an embedding matrix** aligned to a `Vocabulary` (a GloVe vector per
  known word, a random vector for out-of-vocabulary words, zeros for padding)
- **initialize a model's embedding** from that matrix, optionally frozen
- re-benchmark IMDB with GloVe and record the (expected higher) numbers

**Proof of done:** a model can be initialized from GloVe vectors; offline tests
cover loading and alignment with a tiny fake GloVe file; and the README benchmark
is re-run with GloVe (new numbers, expected to break ~86%).

---

# Non-Goals (deferred)

- **Training embeddings** (word2vec / GloVe training) — we consume pretrained
  vectors, not learn them.
- **Contextual embeddings** (ELMo / BERT-style) — much larger; post-1.0.
- **Bundling GloVe files** — they are large (100+ MB); the user downloads them and
  points the example at the path. Tests use a tiny synthetic file.

---

# Abstraction decisions

- A concrete **loader** and **matrix builder** plus a model parameter. No new
  protocol (one embedding source). GloVe requires **word-level** tokenization, so
  it pairs with the whitespace tokenizer, not BPE.

---

# Directory Structure

```
polaris/
│
├── embeddings/
│   ├── __init__.py
│   └── glove.py        # load_glove, build_embedding_matrix
│
└── models/             # + pretrained_embeddings / freeze_embeddings on the
                        #   classifiers

tests/
└── unit/embeddings/
    └── test_glove.py
```

---

# Components (in build order)

1. **`load_glove(path)`** — parse a GloVe text file (`word v1 v2 …` per line) into
   a mapping from word to vector.
2. **`build_embedding_matrix(vocabulary, vectors, *, embedding_dim, seed)`** — a
   `(vocab_size, embedding_dim)` tensor: the GloVe vector for each known token, a
   seeded random vector for out-of-vocabulary tokens, zeros for the padding token.
3. **Model integration** — `pretrained_embeddings` and `freeze_embeddings`
   parameters on `MeanPoolingClassifier` and `TransformerEncoderClassifier`;
   when given, the embedding is initialized from the matrix (via
   `nn.Embedding.from_pretrained`).
4. **Example & re-benchmark** — a `GLOVE_PATH` setting; with a whitespace
   tokenizer and GloVe, build the matrix, initialize the model, train, and record
   the benchmark.

---

# Design Principles

- **Consume, don't train.** Pretrained vectors are an input; the model is still
  ours.
- **Honest coverage.** Out-of-vocabulary words get random (not zero) vectors so
  they are still learnable; only padding is zero.
- **No bundled data.** Large vector files stay out of the repo; tests use a
  synthetic fixture.

---

# Testing Strategy

Offline, with a tiny synthetic GloVe file in a temp directory:

- `load_glove` parses words and vectors correctly.
- `build_embedding_matrix` places each known word's vector on the right row,
  gives out-of-vocabulary tokens non-zero vectors, and the padding row zeros; the
  shape is `(vocab_size, embedding_dim)`.
- A model initialized from a matrix carries those weights; `freeze_embeddings`
  makes the embedding non-trainable.

---

# Deliverables

- `load_glove` and `build_embedding_matrix`
- `pretrained_embeddings` / `freeze_embeddings` on the classifiers
- A GloVe-enabled example and a recorded benchmark
- Complete offline tests, documentation, green CI

---

# What comes next

v0.11 (Deployment & CLI) exposes trained models through a matured `polaris` CLI
and serving.
