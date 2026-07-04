# Phase 07 — Evaluation Framework

**Status:** ✅ Completed

---

# Overview

v0.4.1 added the classification-metric *primitives* — `accuracy`,
`confusion_matrix`, and per-class `precision_recall_f1` — as separate functions,
and the IMDB example hand-assembles a report from them. v0.7 turns those
primitives into a proper, reusable **evaluation framework**: one
`ClassificationReport` value object that captures the whole picture (per-class
and averaged precision/recall/F1, support, accuracy, and the confusion matrix)
with a readable text rendering, plus a harness that evaluates a model on a
dataset and returns it.

This is the rigorous measurement layer a benchmark rests on. (The benchmark
itself goes in the README at v0.8, once experiment tracking makes a run
reproducible.)

---

# Goals

At the end of Phase 7 Polaris can:

- produce a `ClassificationReport` with per-class **and** macro/weighted-averaged
  precision, recall, and F1, plus support, accuracy, and the confusion matrix
- render that report as a readable table (`to_text`)
- evaluate a model on a set of batches and get a report back, decoupled from
  training

**Proof of done:** the IMDB example prints a full classification report, and the
report's values are verified against a hand-computed case.

---

# Non-Goals (deferred)

- **Generation / retrieval metrics** (BLEU, ROUGE, recall@k, …) — there is no
  such task yet. They arrive when a generation/retrieval model does (Post-1.0).
- **Plots / images** (training curves, confusion-matrix heatmaps) — these need a
  plotting dependency (matplotlib). The text report is the priority; plotting is
  optional and deferred (roadmap v0.8+/visualization).
- Multi-label classification, calibration, threshold tuning.

---

# Abstraction decisions

- A **concrete `ClassificationReport` value object** plus functions — no new
  protocol (there is one report type). It **reuses** the v0.4.1 primitives
  (`confusion_matrix`, `precision_recall_f1`, `predict`, `accuracy`) rather than
  reimplementing them.

---

# Directory Structure

```
polaris/
│
└── evaluation/
    ├── metrics.py    # primitives (v0.4.1): accuracy, confusion_matrix, ...
    └── report.py     # ClassificationReport, classification_report, evaluate_model

tests/
└── unit/evaluation/
    └── test_report.py
```

---

# Components (in build order)

1. **`ClassificationReport`** — a frozen value object: `class_names`, per-class
   `precision` / `recall` / `f1` / `support`, overall `accuracy`, `macro_avg`
   and `weighted_avg` of the three metrics, and the `confusion` matrix (as nested
   ints). A `to_text()` renders a readable table.
2. **`classification_report(logits, labels, *, num_classes, class_names=None)`**
   — builds the report from the metric primitives (macro = mean over classes;
   weighted = support-weighted mean).
3. **`evaluate_model(model, batches, *, num_classes, class_names=None)`** — a
   harness that runs `predict` over the batches and returns a
   `ClassificationReport`. Decoupled from training.
4. **Wire into the example** — the IMDB example prints
   `evaluate_model(...).to_text()` instead of hand-assembling metrics.

---

# Design Principles

- **Reuse, don't reimplement.** The report is composed from the existing
  primitives.
- **A value object, not a printout.** The report is structured data first; text
  rendering is one view of it (so it could later be logged, serialized, or
  plotted).
- **Classification only, for now.** We have a classification task; metrics for
  tasks we do not have yet are deferred.

---

# Testing Strategy

Offline, on hand-computable fixtures:

- Per-class and averaged precision/recall/F1, support, accuracy, and confusion
  match a known logits/labels case.
- Macro vs weighted averaging differ correctly on an imbalanced fixture.
- `to_text` contains the class names and the accuracy.

---

# Deliverables

- `ClassificationReport` value object with `to_text`
- `classification_report` and `evaluate_model`
- The example prints a full report
- Complete offline tests, documentation, green CI

---

# What comes next

v0.8 adds experiment tracking and reproducible, config-snapshotted runs — the
point at which a tuned result can be stated as a **benchmark in the README** with
a config + seed to reproduce it.
