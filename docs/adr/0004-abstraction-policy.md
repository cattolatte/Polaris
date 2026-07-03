# ADR 0004 — Evidence-driven abstraction policy

**Status:** Accepted (2026-07-03)

## Context

The ROADMAP already says "abstractions are extracted from working code, never designed
in advance." But the principle was being violated in practice (see ADR-0005 on the
Registry), and "when exactly is an abstraction justified?" was left to judgment. A
concrete, enforceable rule prevents speculative interfaces — the most common way this
kind of project accumulates dead complexity.

## Decision

An abstraction — a `Protocol`, base class, generic, or extension point — is justified
**only when two or more concrete implementations already exist and demand it** (the
"N ≥ 2" rule).

- One implementation → a plain concrete class. No interface.
- A second implementation appears → *now* extract the shared `Protocol`, keeping it
  minimal and structural.
- No configuration knobs, hooks, callbacks, or plugin points are added "in case." They
  are added when a concrete caller needs them.

This applies to infrastructure too: configuration, logging, registries, and CLIs are
built in the phase where a real component first needs them — never speculatively.

The existing pattern is the template: `Dataset`/`Tokenizer` protocols exist alongside
real implementations (`IMDBDataset`, `WhitespaceTokenizer`); the `Model` and `Trainer`
abstractions are deliberately **not** written until the second model and second
training use exist (roadmap v0.5–v0.6).

## Consequences

**Good**

- Interfaces are shaped by real usage, so they fit; they rarely need breaking changes.
- `core` stays small and every abstraction has living consumers.
- New readers never encounter an interface with no implementation behind it.

**Tradeoffs / costs**

- Occasionally the first implementation must be lightly refactored when the second
  arrives and the abstraction is extracted. This is cheap and expected — far cheaper
  than maintaining a wrong abstraction designed up front.
- Requires resisting the (strong) temptation to "just add the interface now."

## Alternatives considered

- **Design interfaces up front:** rejected — produces abstractions that don't fit real
  usage and calcify early. This is the failure mode the whole project is built to
  avoid.
