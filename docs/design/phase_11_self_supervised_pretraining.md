# Phase 11 — Self-Supervised Pretraining (Masked Language Modeling)

Status: 🚧 In Progress · Target: **v0.11.0**

> Written just before implementation, per the docs convention. Keep in sync with
> the code it describes.

## Why this phase

The v0.8–v0.10 benchmarks established, honestly, that ~85–86% is the ceiling for
this **class** of model on IMDB. We pulled three plausible levers — a from-scratch
transformer (v0.5), subword tokenization (v0.9), and pretrained GloVe word vectors
(v0.10) — and **every one bounced off ~86%**. The isolated, remaining lever is the
one that actually transformed NLP: **self-supervised contextual pretraining** —
learn language from *unlabeled* text, then fine-tune on the labeled task.

This phase implements that recipe (the BERT idea) **entirely from scratch**:

- Our own transformer (already built in v0.5),
- Our own masked-language-modeling objective (implemented here),
- Our own weights, trained on our own data.

Nothing is downloaded. "Pretraining" here is a **training method**, not a borrowed
model — so Polaris stays 100% from-scratch (see ADR-0001/0003). This is the rare
case where the from-scratch path and the accuracy path are the *same* path.

## What we build (vertical slice)

The whole pretrain → fine-tune thread must be runnable end to end:

```
unlabeled IMDB text ─► MLM masking ─► [ TransformerEncoder trunk ] ─► MLM head ─► CE on masked tokens
                                              │  (pretrained weights)
                                              ▼  transfer
labeled IMDB text  ─► collation ─► [ TransformerEncoder trunk ] ─► pool ─► classifier ─► fine-tune
```

### 1. A `<mask>` special token

MLM replaces some input tokens with a reserved `<mask>` token whose embedding row
the model must learn. `Vocabulary` already models named special tokens (`unk`,
`pad`); we add `mask_token` in exactly the same shape (a named optional field with
a derived `mask_id`), and thread it through `build_vocabulary`. Pairwise-distinct
validation generalizes to all provided specials. BPE mask support is **deferred**
(no consumer — pretraining runs on the whitespace tokenizer; add later if needed,
per ADR-0004).

### 2. MLM masking / collation — `polaris/pretraining/masking.py`

`MaskedLMBatch` (value object): `input_ids`, `attention_mask`, `labels`, where
`labels` is `(batch, seq)` holding the **original** token id at masked positions
and `IGNORE_INDEX` (-100) everywhere else (so the loss ignores unmasked positions).

`mask_tokens(...)` applies the standard BERT scheme to ~15% of real (non-pad)
tokens: of the chosen positions, **80%** become `<mask>`, **10%** a random token,
**10%** are left unchanged — but all chosen positions are supervised. Seeded for
reproducibility.

### 3. Extract the shared encoder trunk — `polaris/models/transformer_encoder.py`

Now there is a **second** consumer of "embedding + positional + encoder blocks →
per-token hidden states": the MLM head needs those hidden states, and so does the
classifier. Two concrete consumers justify extracting the abstraction (ADR-0004).

`TransformerEncoder(nn.Module)` holds the embedding (with optional pretrained
init), positional encoding, dropout, the encoder-block stack, and the final norm;
`forward(batch) -> (batch, seq, embed)` per-token hidden states. This is a
**behavior-preserving refactor** of the existing classifier — its output must be
numerically identical, so the v0.5 tests continue to pass unchanged.

- `TransformerEncoderClassifier` = trunk + mask-aware mean pool + linear head.
- `MaskedLanguageModel` = trunk + linear MLM head → `(batch, seq, vocab)` logits.

### 4. `MaskedLanguageModel` — `polaris/pretraining/model.py`

The trunk plus a linear projection from `embed_dim` to `vocab_size`. Returns
per-token vocabulary logits. (Weight tying with the embedding is a known refinement
we deliberately skip for readability; noted as a possible follow-up.)

### 5. Pretraining loop — `polaris/pretraining/loop.py`

A dedicated `pretrain(...)` — cross-entropy over masked positions only
(`ignore_index=-100`), reusing the v0.6 warmup scheduler. It is deliberately
*separate* from `Trainer`: `Trainer` computes a classification loss and accuracy,
whereas MLM optimizes a token-level objective with a different eval signal
(masked-token accuracy / perplexity). Forcing them into one abstraction would be
speculative generality — we keep two concrete loops (ADR-0004).

### 6. Fine-tuning: transfer the trunk

After pretraining, load the trunk weights into a fresh classifier
(`classifier.encoder.load_state_dict(mlm.encoder.state_dict())`) and fine-tune with
the existing `Trainer`. Because both models embed the *same* `TransformerEncoder`
submodule, the transfer is a plain `state_dict` copy — no surgery.

### 7. Example + benchmark

Extend `examples/train_imdb_sentiment.py` (or a sibling) to: pretrain on the
**unlabeled IMDB split** (50k reviews), then fine-tune on the 25k labeled set, and
report against the from-scratch baseline. This is the run that aims to finally move
the number — honestly measured either way.

## Design principles / non-goals

- **From scratch, no downloads.** We implement the objective and train the weights;
  we never load an external pretrained model. GloVe (v0.10) is the closest we come
  to external data, and pretraining our *own* embeddings makes even that optional.
- **Two concrete loops, not a premature `Trainer` generalization.** Classification
  and MLM differ enough that unifying them now would be speculative (ADR-0004).
- **No weight tying, no NSP, no whole-word masking (yet).** The simplest correct MLM
  that teaches the idea. Refinements are follow-ups with their own evidence.
- **Offline, tiny-fixture tests.** Masking statistics/shape/ignore-index structure;
  MLM output shape `(B, S, vocab)`; trunk output unchanged by the refactor; pretrain
  reduces loss on a tiny corpus; trunk transfer copies weights exactly.

## Module layout

```
polaris/models/transformer_encoder.py   # NEW  TransformerEncoder (shared trunk)
polaris/models/transformer_classifier.py# refactor to use the trunk (behavior-preserving)
polaris/pretraining/__init__.py         # NEW  public surface
polaris/pretraining/masking.py          # NEW  MaskedLMBatch, mask_tokens, IGNORE_INDEX
polaris/pretraining/model.py            # NEW  MaskedLanguageModel
polaris/pretraining/loop.py             # NEW  pretrain(...)
polaris/tokenizers/vocabulary.py        # add mask_token / mask_id
polaris/tokenizers/builder.py           # add mask_token to build_vocabulary
```

Dependency direction stays one-way: `core → tokenizers → collation → models →
pretraining → training`. `pretraining` depends on `models` (the trunk) and
`collation`, never the reverse.
