# Architecture Decision Records

This directory records the **significant, long-lived decisions** behind Polaris —
the ones that would be expensive or confusing to reverse, and that a new contributor
needs the *reasoning* for, not just the outcome.

An ADR is short. It captures the context at the time, the decision, and the
consequences (including the tradeoffs we accepted). ADRs are **immutable once
accepted**: we do not edit history. If a decision changes, write a new ADR that
supersedes the old one and update the old one's status.

## When to write one

Write an ADR when a decision:

- affects the architecture or shapes multiple future phases (e.g. the tensor
  framework, how we decompose work), **or**
- resolves a genuine fork with real tradeoffs, **or**
- would otherwise be made silently and forgotten (and later questioned).

Do **not** write an ADR for routine, local, or easily-reversible choices. Most code
decisions are not ADRs.

## Status values

`Proposed` → `Accepted` → (later, if replaced) `Superseded by ADR-NNNN` /
`Deprecated`.

## Format

Copy an existing ADR. Each has: **Title**, **Status**, **Context**, **Decision**,
**Consequences** (tradeoffs, good and bad), and **Alternatives considered**. Number
files sequentially: `NNNN-kebab-title.md`.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-project-identity.md) | Project identity: educational reference implementation | Accepted |
| [0002](0002-vertical-slice-architecture.md) | Vertical-slice architecture over horizontal layers | Accepted |
| [0003](0003-tensor-framework.md) | PyTorch as the tensor framework; model internals from scratch | Accepted |
| [0004](0004-abstraction-policy.md) | Evidence-driven abstraction policy | Accepted |
| [0005](0005-registry-dormant.md) | Registry and Component are dormant until a consumer exists | Accepted |
