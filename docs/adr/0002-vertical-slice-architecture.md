# ADR 0002 — Vertical-slice architecture over horizontal layers

**Status:** Accepted (2026-07-03)

## Context

The ROADMAP states, in bold, "Vertical slices over infrastructure layers — each
release delivers working functionality, not scaffolding." But the phase structure had
drifted into horizontal layers: all datasets, then all tokenizers, then all models,
then all training, then all evaluation.

That layering has a fatal property: it produces components that cannot run. A "Model
Zoo" phase before a training phase builds models that can't be trained; an evaluation
phase after training means training invents throwaway metrics. By the project's own
definition, unrunnable components are scaffolding. The macro-structure contradicted
the stated philosophy.

## Decision

Structure all future work as **vertical slices**. Every milestone must leave Polaris
in a **runnable** state — a working thread through the layers it touches — rather than
completing one horizontal layer in isolation.

The pivotal instance is the **first end-to-end slice** (roadmap v0.4): the thinnest
runnable thread from raw text to a measured result —
`Dataset → Tokenizer → Collation → Model → Training → Basic Evaluation` on IMDB. Later
phases *thicken* individual layers (a real transformer, a mature trainer, a full
evaluation framework), always keeping the pipeline runnable.

Corollaries:
- Never build a model that cannot be trained, or a trainer with nothing to train.
- Basic, "good enough to run the slice" versions of a layer come first; comprehensive
  versions come later, once the seams are proven.
- Prefer one thin thread through every layer over one thick layer with no thread.

## Consequences

**Good**

- A demoable, runnable artifact arrives early (train a sentiment classifier on IMDB
  from raw text), which is what earns trust, stars, and contributors.
- Integration seams (collation, tensor conversion, config) surface early and cheaply,
  while they're still easy to change.
- Faithful to the philosophy the project already espouses.

**Tradeoffs / costs**

- Each layer is revisited across multiple phases (e.g. the model layer grows in v0.4
  then v0.5). This is intentional iteration, but it means no layer is "finished" in
  one pass.
- Requires discipline to keep early slices genuinely thin and resist front-loading a
  layer "while we're in there."

## Alternatives considered

- **Horizontal layering (status quo):** rejected — produces unrunnable scaffolding and
  delays any working system for many phases.
