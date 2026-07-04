# Phase 06 — Training Engine Maturity

**Status:** ✅ Completed

> Where two deferred pieces finally arrive because a real component needs them:
> the **configuration system** and **logging** (see `ROADMAP.md` and ADR-0004).

---

# Overview

v0.4 and v0.5 train with a deliberately minimal `train()` function — explicit
arguments, no state. That was correct for a thin slice, but it cannot do what
real training needs, and the v0.5 transformer made the gap concrete: it
**underfit** because there is no learning-rate warmup, and we keep the *last*
model rather than the *best*.

v0.6 grows the training layer into a real engine — a `Trainer` that composes a
learning-rate schedule, checkpointing, a validation loop with early stopping,
config-driven setup, and logging — while keeping the pipeline runnable end to
end.

---

# Goals

At the end of Phase 6 Polaris can:

- schedule the learning rate with **warmup** then decay (the transformer's
  missing piece)
- **checkpoint** model + optimizer state, and keep the *best* model by a
  validation metric
- run a **validation loop** with **early stopping**
- drive a run from a **typed configuration** rather than edited constants
- **log** per-epoch metrics through a standard logger

**Proof of done:** the transformer, trained via the `Trainer` with warmup
scheduling and best-checkpointing, beats the v0.4 baseline — and the run is
defined by a config object, not code edits.

---

# Non-Goals (deferred)

- A generic **callback / plugin system** — early stopping and logging are built
  into the `Trainer`; a `Callback` protocol waits until a real third-party hook
  needs one (the consumer rule).
- Distributed / multi-GPU training, mixed precision — Post-1.0.
- Experiment tracking (MLflow / TensorBoard), config snapshots — v0.8.
- Hyperparameter search / sweeps — later.

---

# Abstraction decisions (applying ADR-0004 / ADR-0005)

- **`Trainer` — justified now.** It is the *vehicle* that holds the cross-epoch
  state the new features need (the schedule, the best-so-far checkpoint, the
  early-stopping counter) and it has a consumer (the example / CLI). The minimal
  `train()` function stays for the simple case and the learning-behaviour tests.
- **Configuration — a concrete `TrainingConfig`, not a framework.** A typed
  Pydantic model of the hyperparameters, loadable from a file. It grows fields
  when a run needs them — no generic config engine.
- **Callbacks — deferred.** No consumer needs a generic hook yet; extracting a
  `Callback` protocol now would be a speculative dead abstraction (ADR-0005).
- **Scheduler — from scratch.** A small warmup-then-decay schedule, because
  warmup is a concept worth showing, not hiding behind a library call (ADR-0003).

---

# Directory Structure

```
polaris/
│
├── training/
│   ├── __init__.py
│   ├── loop.py          # existing minimal train() (kept)
│   ├── scheduler.py     # WarmupSchedule: warmup then decay (from scratch)
│   ├── checkpoint.py    # save/load model + optimizer + metadata
│   ├── config.py        # TrainingConfig (Pydantic)
│   └── trainer.py       # Trainer: composes the above
│
└── utils/
    └── logging.py       # get_logger(...)

tests/
└── unit/training/
    ├── test_scheduler.py
    ├── test_checkpoint.py
    ├── test_config.py
    └── test_trainer.py
```

---

# Components (in build order)

1. **Learning-rate schedule** (`scheduler.py`) — a from-scratch warmup-then-decay
   schedule: the LR ramps linearly from ~0 to the peak over `warmup_steps`, then
   decays (linear or cosine) to ~0 over the remaining steps. Applied to the
   optimizer each step. Tested: near-zero at step 0, peak right after warmup,
   monotone decay afterward.
2. **Checkpointing** (`checkpoint.py`) — save `{model_state, optimizer_state,
   metadata}` to a path and restore it. Tested: a round-trip restores the model's
   weights exactly (offline, into a temp dir).
3. **Logging** (`utils/logging.py`) — a small `get_logger` returning a configured
   standard-library logger. The first concrete home for logging.
4. **`TrainingConfig`** (`config.py`) — a Pydantic model of the run
   hyperparameters (epochs, LR, warmup, batch size, early-stopping patience, …),
   with validation and defaults; constructable from a dict / file. Tested:
   validation and defaults.
5. **`Trainer`** (`trainer.py`) — composes model, optimizer, schedule,
   checkpointing, and logging; runs train + validation epochs, applies early
   stopping, and keeps the best model by validation metric. Tested on tiny
   synthetic data: loss decreases, early stopping triggers, the best checkpoint
   is written.
6. **Wire into the example** — a config-driven transformer run with a validation
   split, warmup scheduling, and best-checkpoint restore.

---

# Design Principles

- **Infrastructure when needed.** Config and logging appear now, because the
  `Trainer` is the first component that genuinely needs them.
- **From scratch where it teaches.** The warmup schedule is hand-written.
- **Keep it runnable.** Every sub-slice merges green; the transformer keeps
  training throughout.
- **No speculative extension points.** Built-in early stopping and logging, not a
  generic callback system.

---

# Testing Strategy

Offline, tiny fixtures, behaviour not floats:

- Scheduler: LR profile (warmup up, then down); values at known steps.
- Checkpoint: save→load restores weights (temp dir, no network).
- Config: validation errors and defaults.
- Trainer: loss decreases on synthetic data; early stopping halts; best
  checkpoint is saved and restorable.

---

# Deliverables

- Warmup learning-rate schedule
- Checkpoint save / restore
- `TrainingConfig` (Pydantic)
- `Trainer` with validation, early stopping, best-checkpointing, logging
- A config-driven transformer example
- Complete offline tests, documentation, green CI

---

# What comes next

v0.7 builds the comprehensive evaluation framework on top of the basic metrics;
v0.8 adds experiment tracking and reproducible, config-snapshotted runs — the
point at which a benchmark can be stated in the README with a config to
reproduce it.
