# Phase 12 — Deployment & CLI

Status: 🚧 In Progress · Target: **v0.12.0** (final feature phase before v1.0)

> Written just before implementation, per the docs convention. Keep in sync with
> the code it describes.

## Why this phase

Polaris can train, evaluate, and pretrain models — but only from inside a Python
script. A trained model has no life outside the process that made it. This phase
closes the last gap in the end-to-end story: **make a trained Polaris model a
usable artifact** — save it, reload it anywhere, and run predictions on raw text
from the command line (and, as a second slice, over HTTP).

This is the milestone that turns "a codebase you read" into "a system you run."

## The gap in one sentence

`save_checkpoint` (v0.6) persists *weights only* — you must already have the exact
model object constructed to load them back. Inference needs the opposite: a
**self-describing bundle** that carries everything required to rebuild the model
*and* tokenize new text, so a caller who has never seen the training code can load
it and predict.

## Slice 1 — Inference bundle + `polaris predict` (the core runnable slice)

A new `polaris/inference/` module.

### The bundle

A single file (`torch.save` payload) that is fully self-contained:

```
model_type      : "pooling" | "transformer"
model_config    : the exact kwargs the model was built with (vocab_size, dims, …)
model_state     : the trained state_dict
vocabulary      : token_to_id + special tokens (JSON-able)
label_names     : e.g. ["neg", "pos"]
polaris_version : provenance
```

- `save_bundle(path, *, model, tokenizer, model_type, model_config, label_names)`
- `load_bundle(path) -> Predictor`

`Vocabulary` gains `to_dict` / `from_dict` (a pure, validated round-trip) so it can
be serialized — the value object owns its own serialization, mirroring how it owns
its own validation.

### The model factory

`build_model(model_type, model_config) -> nn.Module` — a concrete `match` on the
two model types. **The dormant registry stays dormant** (ADR-0005): two model
types and a single consumer do not justify a plugin lookup; a `match` statement is
more readable for a learner and has no indirection. This is documented as a
deliberate call; the registry becomes justified only when model types proliferate
*and* several subsystems need name-based lookup.

### The predictor

`Predictor` wraps a model (in `eval` mode) and its tokenizer, and owns the raw-text
path end to end: `tokenize → encode → collate → forward → softmax → label`.

```python
predictor = load_bundle("model.pt")
predictor.predict("a wonderful, moving film")
# Prediction(label="pos", label_id=1, probabilities={"neg": 0.02, "pos": 0.98})
```

- `predict(text: str) -> Prediction` and `predict_batch(texts) -> list[Prediction]`.
- `Prediction` is a frozen value object (label, label_id, probabilities).
- Whitespace tokenizer only in this slice (it backs the benchmarks). BPE bundles
  need merge serialization — deferred until there is a consumer (ADR-0004).

### The CLI

Grow the Typer app (currently just `info`) with:

- `polaris predict --model PATH TEXT` — print the predicted label (and
  probabilities with `--probs`). Reads `TEXT` from the argument or stdin.

A training example gains a `save_bundle` call so there is a real artifact to
predict from; the offline tests train a tiny model, round-trip a bundle, and assert
predictions — no network.

## Slice 2 — Evaluate command + HTTP serving (same release, second thread)

- `polaris evaluate --model PATH --split test` — load a bundle, evaluate on IMDB,
  print the classification report (reusing the v0.7 evaluation framework).
- `polaris/deployment/` (currently a stub): a minimal **FastAPI** app exposing
  `POST /predict` over the `Predictor`, imported lazily (own-the-interface: FastAPI
  is an optional extra, wrapped, never leaking into core). A short Dockerfile.

ONNX export is explicitly **out of scope / best-effort** — kept off the critical
path, per the roadmap.

## Design principles / non-goals

- **Own the interface.** FastAPI and any serving deps are optional extras, imported
  lazily, and failures wrap in a `PolarisError`. The bundle format is Polaris-native.
- **Concrete before abstract.** A `match`-based model factory, not a reactivated
  registry. No generic "serialization framework."
- **Vertical slice.** Slice 1 alone leaves Polaris runnable as a product (train →
  save → predict from the shell). Slice 2 adds serving on top of the same
  `Predictor`; it reuses, it does not rebuild.
- **Offline tests.** Bundle round-trip, factory reconstruction, and prediction on
  tiny fixtures; the FastAPI app tested with its `TestClient`, no network.

## Module layout

```
polaris/inference/__init__.py        # NEW  public surface
polaris/inference/bundle.py          # NEW  save_bundle / load_bundle
polaris/inference/factory.py         # NEW  build_model (match on model_type)
polaris/inference/predictor.py       # NEW  Predictor, Prediction
polaris/tokenizers/vocabulary.py     # add Vocabulary.to_dict / from_dict
polaris/cli.py                       # add `predict` (and later `evaluate`)
polaris/deployment/app.py            # NEW (slice 2)  FastAPI POST /predict
```

Dependency direction stays one-way: `inference` depends on `models`, `tokenizers`,
`collation`; `deployment` depends on `inference`; the CLI depends on all. Nothing
depends back on the CLI or deployment.
