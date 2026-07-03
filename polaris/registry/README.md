# Registry Module

Provides `Registry`, an in-memory, instance-based store for discovering Polaris
components by name (`register`, `get`, `contains`, `remove`, `clear`, `list`), with
no global mutable state.

## Status: dormant

This module is **implemented and tested, but currently has no consumers** — nothing
in Polaris registers itself or exposes the `Component.metadata` the registry is
designed to hold.

Per [ADR-0005](../../docs/adr/0005-registry-dormant.md), the registry is **parked as
dormant**: do not build new features on it, and do not force other types to implement
`Component` just to populate it. It will be reactivated when a real consumer appears
(most likely experiment tracking or the CLI needing name-based lookup), at which point
its design will be validated against that need.
