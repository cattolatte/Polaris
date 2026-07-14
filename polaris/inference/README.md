# Inference Module

Turns a trained Polaris model into a **usable artifact**. Training produces a model
object inside a script; this module lets you save it as a self-contained *bundle*,
reload it anywhere, and classify raw text — with no training code in sight. It is
the last link in the end-to-end story (data → … → training → **inference**).

## Why a bundle (and not just a checkpoint)

A training checkpoint (`polaris.training.save_checkpoint`) stores **weights only** —
you must already have the exact model object constructed to load them back. A
*bundle* is **self-describing**: it also carries the model type, its constructor
arguments, the vocabulary, and the class names, so a caller who has never seen the
training code can reload it and predict.

## Public surface

- `save_bundle` / `load_bundle` (`bundle.py`) — write a bundle (a single
  `torch.save` payload of Polaris-native parts) and load it back as a ready
  `Predictor`. Whitespace tokenizer only for now (BPE bundles need merge
  serialization; deferred until there is a consumer).
- `Predictor` / `Prediction` (`predictor.py`) — `Predictor` owns the raw-text path
  (`tokenize → encode → collate → forward → softmax → label`) in `eval` mode under
  `torch.no_grad`; `predict` / `predict_batch` return `Prediction` value objects
  (label, label id, per-class probabilities).
- `build_model` (`factory.py`) — reconstruct a model from its saved type and
  config. A concrete `match` over the model types, **not** the dormant registry
  (ADR-0005): two types and one consumer do not justify a plugin lookup.
- `encode_texts` (`embedding.py`) — batch-embed a sequence of texts with a
  `TextEmbedder`, returning a `(len(texts), embedding_dim)` NumPy array (the
  corpus-embedding step of a dense retriever).

## Example

```python
from polaris.inference import save_bundle, load_bundle

# After training `model` with `tokenizer`:
save_bundle(
    "model.pt",
    model=model,
    tokenizer=tokenizer,
    model_type="transformer",
    model_config={"num_classes": 2, "vocab_size": len(vocab), "embed_dim": 128, ...},
    label_names=["neg", "pos"],
    max_length=256,
)

predictor = load_bundle("model.pt")
predictor.predict("a wonderful, moving film")
# Prediction(label="pos", label_id=1, probabilities={"neg": 0.02, "pos": 0.98})
```

Or from the shell:

```bash
polaris predict "a wonderful, moving film" --model model.pt --probs
```
