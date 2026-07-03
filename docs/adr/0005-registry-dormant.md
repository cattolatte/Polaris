# ADR 0005 — Registry and Component are dormant until a consumer exists

**Status:** Accepted (2026-07-03)

## Context

`polaris/registry/` (a `Registry`) and `polaris/core/` (`Component`,
`ComponentMetadata`) were built during the v0.1 Foundation phase. They are real,
typed, and tested — but they have **no consumers**. No `Dataset` or `Tokenizer`
implements `Component`, exposes `.metadata`, or registers itself. The Registry is
speculative infrastructure built ahead of need, which directly contradicts the
project's evidence-driven abstraction policy (ADR-0004). Leaving a dead abstraction in
`core` while preaching "concrete before abstract" is the most visible
philosophy/implementation contradiction in the repo.

Options weighed: wire it in now, remove it entirely, leave it unused, or park it.

## Decision

**Park the Registry and Component abstraction as dormant.** Keep the existing tested
code, but treat it as inactive:

- Do **not** build new features on it.
- Do **not** force `Dataset`, `Tokenizer`, model, or trainer types to implement
  `Component` or expose `.metadata` just to give the Registry something to hold.
- Reintroduce it **only when a real consumer appears** — the most likely first
  consumer is experiment tracking or the CLI needing name-based component discovery
  (roadmap ~v0.8–v0.9). At that point, validate the design against the real need and
  wire it in (or revise it) then.
- `polaris/registry/README.md` notes this dormant status and points here.

## Consequences

**Good**

- Resolves the contradiction without discarding correct, tested, git-tracked work.
- Preserves the reasoning and the code for when a genuine need arrives, avoiding a
  from-scratch rebuild.
- Keeps the near-term surface honest: nothing new depends on an unproven abstraction.

**Tradeoffs / costs**

- A small amount of unused code remains in the tree. Mitigated by the explicit dormant
  marker and this ADR, so it is not mistaken for active, load-bearing infrastructure.
- When reactivated, the design may need revision to fit the real consumer — meaning
  some of the current API may change. That is acceptable and expected.

## Alternatives considered

- **Wire it in now:** rejected — forces `Component`/metadata onto every type before
  any need exists, spreading speculative coupling; contradicts ADR-0004.
- **Remove it entirely:** rejected — discards working, tested code and its history for
  a subsystem we will very likely want by v0.8–v0.9.
- **Leave as-is, unmarked:** rejected — leaves the contradiction in `core` with no
  signal that the code is inactive.
