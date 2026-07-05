# Models Module

The neural models, with all model-defining internals **implemented from scratch** on
PyTorch primitives (ADR-0003): our own attention, transformer blocks, layer norm,
and positional encoding. The framework supplies autograd and parameter containers —
nothing that makes the model *a transformer* is imported.

Every model consumes a `Batch` and returns logits, so they are drop-in
interchangeable in the same training loop.

## Public surface

- `MeanPoolingClassifier` (`pooling_classifier.py`) — the v0.4 baseline: embed,
  mask-aware mean-pool over real tokens, linear head. The simplest thing that proves
  the pipeline.
- `TransformerEncoder` (`transformer_encoder.py`) — the shared, **headless** trunk:
  embedding + sinusoidal positional encoding + a stack of encoder blocks + final
  norm, producing per-token hidden states. Reused by both the classifier and the
  masked-language model, so a pretrained trunk transfers by a `state_dict` copy.
- `TransformerEncoderClassifier` (`transformer_classifier.py`) — the trunk plus
  mask-aware mean pooling and a linear head (v0.5).

Both classifiers accept `pretrained_embeddings` (optionally frozen).

## From-scratch building blocks

Not exported at the package top, but the heart of the module:

- `MultiHeadSelfAttention` (`attention.py`) — scaled dot-product attention with
  padding masks, split across heads.
- `LayerNorm`, `SinusoidalPositionalEncoding`, `TransformerEncoderBlock`
  (`transformer.py`) — a pre-norm encoder block (attention + FFN, each residual +
  normalized), from scratch.
