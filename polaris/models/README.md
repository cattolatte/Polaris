# Models Module

The neural models, with all model-defining internals **implemented from scratch** on
PyTorch primitives (ADR-0003): our own attention, transformer blocks, layer norm,
and positional encoding. The framework supplies autograd and parameter containers —
nothing that makes the model *a transformer* is imported.

The classifiers consume a `Batch` and return logits; the embedder returns a
pooled vector. They share the same trunk, so they are interchangeable heads.

## Public surface

- `MeanPoolingClassifier` (`pooling_classifier.py`) — the v0.4 baseline: embed,
  mask-aware mean-pool over real tokens, linear head. The simplest thing that proves
  the pipeline.
- `TransformerEncoder` (`transformer_encoder.py`) — the shared, **headless** trunk:
  embedding + sinusoidal positional encoding + a stack of encoder blocks + final
  norm, producing per-token hidden states. Reused by the classifier, the
  masked-language model, and the embedder, so a pretrained trunk transfers by a
  `state_dict` copy.
- `TransformerEncoderClassifier` (`transformer_classifier.py`) — the trunk plus
  mask-aware mean pooling and a linear head (v0.5).
- `TextEmbedder` (`embedder.py`) — the bi-encoder tower (v1.2): trunk +
  `mean_pool` + optional linear projection + optional L2-normalization, emitting a
  single embedding per text. `forward(Batch)` and `encode(input_ids, attention_mask)`.
  Train it with the InfoNCE objective (`polaris.training.losses`).
- `SentencePairClassifier` (`pair_classifier.py`) — the cross-encoder (v1.3):
  joint-encode `[CLS] a [SEP] b [SEP]`, pool (`"cls"`/`"mean"`), and score with a
  linear head. `num_classes` selects rerank (1), gate (2), or NLI (3).
- `mean_pool` (`pooling.py`) — the mask-aware mean over real tokens, shared by both
  classifiers and the embedder.
- `HasEncoder` (`transformer_encoder.py`) — the structural type of "a model with a
  `TransformerEncoder` as `encoder`", so a pretrained trunk transfers between heads.

Both classifiers and the embedder accept `pretrained_embeddings` (optionally frozen).

## From-scratch building blocks

Not exported at the package top, but the heart of the module:

- `MultiHeadSelfAttention` (`attention.py`) — scaled dot-product attention with
  padding masks, split across heads.
- `LayerNorm`, `SinusoidalPositionalEncoding`, `TransformerEncoderBlock`
  (`transformer.py`) — a pre-norm encoder block (attention + FFN, each residual +
  normalized), from scratch.
