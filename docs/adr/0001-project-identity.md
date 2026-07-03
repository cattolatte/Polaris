# ADR 0001 — Project identity: educational reference implementation

**Status:** Accepted (2026-07-03)

## Context

Polaris could pursue two identities that pull design in opposite directions:

1. A **usable NLP platform** competing on features with Hugging Face + PyTorch
   Lightning + MLflow.
2. An **educational, reference implementation** — a from-scratch NLP system whose
   value is the clarity and engineering quality of the codebase itself.

These are not fully compatible. Chasing (1) means breadth, third-party wrapping, and
feature parity we cannot win against incumbents. Everything about how Polaris is
already built — strict typing, per-phase design docs, "own the interface,"
evidence-driven abstraction, "Educational" listed as a stated goal — points at (2).
An ambiguous identity was producing roadmap decisions optimized for (1)'s breadth
while the philosophy optimized for (2)'s depth.

## Decision

Polaris is an **educational, reference-implementation-first NLP engineering
platform.** The primary product is the codebase: readable, typed, tested, documented,
and integrated well enough to run a real task end-to-end and reproducibly.

Concretely, this identity is the tie-breaker for design decisions:

- Prefer the implementation a reader **learns the most** from, provided it is correct,
  clean, and well-engineered.
- Do **not** add features to chase parity with incumbents. When "more features"
  conflicts with "clearer and better-engineered," clarity wins.
- Favor completeness of a **thin end-to-end story** over breadth of half-built layers.

## Consequences

**Good**

- A clear north star that resolves scope and ordering disputes.
- A defensible niche: "the clearest well-engineered from-scratch NLP system" is a
  category Polaris can lead, unlike "a worse Hugging Face."
- Strong portfolio / open-source signal: demonstrates engineering *judgment*, not just
  the ability to call a library.

**Tradeoffs / costs**

- Polaris will deliberately lack features real users might want (broad model zoo,
  distributed training, many datasets). That is accepted.
- "Educational value" is somewhat subjective and must be applied with taste, not as a
  license to over-explain or gold-plate.
- Some engineering (from-scratch internals) costs more effort than wrapping a library
  would — justified by the learning payoff.

## Alternatives considered

- **Usable platform (identity 1):** rejected — unwinnable against incumbents, and it
  contradicts the existing engineering philosophy.
- **Leave identity implicit:** rejected — the ambiguity was actively causing
  misaligned roadmap decisions.
