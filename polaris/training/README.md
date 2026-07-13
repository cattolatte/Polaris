# Training Module

The engine that fits a model to data — from a minimal loop to a full training
engine, plus specialized objectives that don't fit the plain classifier shape.

## Public surface

- `train` (`loop.py`) — the minimal loop: for each epoch/batch, forward → loss →
  backward → step; returns per-epoch mean loss. Takes an optional `loss_fn`.
- `Trainer` (`trainer.py`) — the training engine: validation, early stopping,
  best-model tracking, logging, and checkpointing, driven by a `TrainingConfig`.
  Returns a `TrainingResult` (a history of `EpochRecord`s).
- `TrainingConfig` (`config.py`) — a Pydantic model of the run settings, with
  `to_file` / `from_file`.
- `WarmupSchedule` (`scheduler.py`) — linear warmup then linear/cosine decay,
  from scratch.
- `save_checkpoint` / `load_checkpoint` (`checkpoint.py`) — persist and restore
  model (and optional optimizer) state plus small metadata.

### Contrastive training (v1.2)

- `info_nce_loss` (`losses.py`) — the InfoNCE objective for a bi-encoder: in-batch
  negatives, optional per-anchor hard negatives, optional symmetric term. Operates
  on two L2-normalized embedding tensors (not `(logits, labels)`), so it lives here
  as a function rather than an `nn.Module` criterion.
- `train_contrastive` (`contrastive.py`) — a minimal seeded driver that trains a
  `TextEmbedder` on `ContrastiveBatch`es with `info_nce_loss`.

## Design note

Distinct training paths (classifier `train`/`Trainer`, MLM `pretrain`, contrastive
`train_contrastive`) are kept as small concrete loops rather than one generalized
abstraction — they optimize genuinely different objectives (ADR-0004).
