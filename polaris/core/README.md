# Core Module

The foundation every other Polaris module builds on: shared **types** and the
behavioral **interfaces** (protocols) that define the system's contracts. It has no
dependencies on other Polaris modules — the dependency arrow points *into* `core`,
never out of it.

## Public surface

- **Types** (`types.py`) — the shared vocabulary: `DatasetSplit`
  (`"train" | "test" | "unsupervised"`), `Identifier`, `Version`, `Tag`, and the
  `ComponentType` / `ComponentMetadata` used by the (dormant) registry.
- **Interfaces** (`interfaces.py`) — `@runtime_checkable` `Protocol`s that state
  what a component must *do*, with no implementation. Following the abstraction
  policy (ADR-0004), a protocol is extracted only once two concrete implementations
  prove its shape.

## Design notes

- `core` is pure and framework-free: no PyTorch, no I/O. Importing it is cheap and
  has no optional-dependency cost.
- Types are modern Python 3.12 (`type` aliases, `StrEnum`, `X | None`).
