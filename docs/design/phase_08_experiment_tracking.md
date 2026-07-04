# Phase 08 — Experiment Tracking & Reproducibility

**Status:** 🚧 In Progress

---

# Overview

Polaris can now train (v0.6) and measure (v0.7), but a run is not yet
*reproducible* or *comparable*: the hyperparameters live in a config object that
vanishes when the process ends, and nothing records the environment or the exact
result. v0.8 fixes that. It makes a run a durable, reproducible artifact —
config snapshots, an environment/seed record, and a local experiment tracker that
writes each run (config + metric history + classification report + environment)
to disk.

This is the phase we have been building toward for the **README benchmark**:
once a run is reproducible and recorded, a result can be stated with a config and
seed to reproduce it.

This phase also closes two threads **carried from v0.6** (see the correction note
in `phase_06_training_engine.md`):

1. **`TrainingConfig` load/save from a file** — deferred here because config
   *snapshots* are the point of this phase.
2. **Actually demonstrating the transformer beating the baseline** — via a
   reproducible, full-data run recorded by the tracker, written into the README.

---

# Goals

At the end of Phase 8 Polaris can:

- **snapshot** a `TrainingConfig` to and from a file
- **capture** the environment (Python / PyTorch / Polaris versions, platform) and
  the seed
- **record** a run — its config, metric history, and classification report, plus
  the environment — into a run directory, and read it back
- state a **reproducible benchmark** in the README, backed by a recorded run

**Proof of done:** a tracked run directory exists containing the config, metrics,
report, and environment; a config round-trips to/from a file; and the README has
a **Results** section citing a number produced by a recorded run, with the config
and seed to reproduce it.

---

# Non-Goals (deferred)

- **MLflow / TensorBoard / cloud backends** — one local filesystem backend first
  (concrete before abstract); a second backend waits for a real need.
- Run-comparison UIs, dashboards, hyperparameter sweeps — Post-1.0.
- Distributed artifact stores.

---

# Abstraction decisions (applying ADR-0004 / ADR-0005)

- **One local backend, concrete.** The tracker writes JSON + text to a run
  directory. No tracking *protocol* yet — there is one backend, so an interface
  would be speculative.
- **The Registry stays dormant.** The roadmap once pencilled in "reconsider the
  Registry here," but experiment tracking does **not** need name-based component
  discovery — no consumer appears. ADR-0005 holds; we do not reactivate it. (The
  v0.9 CLI is the next candidate consumer.)

---

# Directory Structure

```
polaris/
│
├── training/
│   └── config.py            # + to_file / from_file (config snapshots)
│
└── experiments/
    ├── __init__.py
    ├── environment.py       # capture_environment()
    └── tracker.py           # record_run(...) / load_run(...)

tests/
└── unit/experiments/
    ├── test_environment.py
    └── test_tracker.py
```

---

# Components (in build order)

1. **Config snapshots** — `TrainingConfig.to_file(path)` / `TrainingConfig.from_file(path)`
   (JSON via Pydantic). Closes carried thread (1). Tested: round-trip equality.
2. **Environment capture** — `capture_environment()` returning the Python,
   platform, PyTorch, and Polaris versions. Tested: expected keys present.
3. **Experiment tracker** — `record_run(directory, *, config, history, report,
   environment, seed)` writes a run directory (`config.json`, `metrics.json`,
   `report.txt`, `environment.json`, `run.json`); `load_run(directory)` reads the
   structured parts back. Tested offline in a temp dir: write then read.
4. **Reproducible benchmark** — a config (full data, tuned) for the transformer
   and the baseline; a recorded run for each; and a **Results** section in the
   README citing the numbers with the config + seed to reproduce them. Closes
   carried thread (2) and the roadmap's README-benchmark goal.

---

# Design Principles

- **A run is an artifact.** Config, metrics, report, and environment are written
  to disk so a result can be reproduced and compared.
- **One backend, concrete.** No tracking abstraction until a second backend earns
  it.
- **Reproducibility is captured, not assumed.** Seed and environment are recorded
  with every run.

---

# Testing Strategy

Offline, temp directories, no network:

- Config round-trips to and from a file (equality preserved).
- `capture_environment` returns the expected keys with plausible values.
- `record_run` writes the artifacts; `load_run` reads the config, metrics, and
  environment back.

(The benchmark run itself is executed by hand on the full dataset — its recorded
numbers, not the training, are what the README cites.)

---

# Deliverables

- `TrainingConfig` file snapshots (`to_file` / `from_file`)
- `capture_environment`
- `record_run` / `load_run` (local experiment tracker)
- A **Results** section in the README backed by a reproducible, recorded run
- Complete offline tests, documentation, green CI

---

# What comes next

v0.9 (Deployment & CLI) exposes trained models through a matured `polaris` CLI
and serving — and is the next candidate consumer for the dormant Registry.
