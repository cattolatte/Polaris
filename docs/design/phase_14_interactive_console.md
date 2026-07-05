# Phase 14 — Interactive Console

Status: 🚧 In Progress · Target: **v1.1.0** (first post-stable feature)

## Why

The CLI is one-shot: every `polaris predict` pays the full model-load cost and
exits. An **interactive console** (the `msfconsole` pattern: a banner, a `polaris >`
prompt, `help`) loads a bundle once and predicts many times instantly — better for
demos and hands-on exploration, and a natural fit for a teaching platform.

## Design

- **`polaris console`** launches a REPL built on the standard library's `cmd.Cmd` —
  the textbook module for exactly this (prompt loop, `do_<command>` methods,
  built-in `help`). **Zero new dependencies.**
- **Banner:** one curated block-letter banner (user-selected), colored by a
  **randomly chosen ANSI color scheme per launch** — the Metasploit approach
  (bundled art + randomness), no network, no API. Color is suppressed when stdout
  is not a TTY or `NO_COLOR` is set (https://no-color.org).
- **Commands:** `load <bundle>`, `predict <text>`, `probs` (toggle probability
  display), `info`, `exit`/`quit` (+ `help`, built in). `predict` before `load`
  prints a friendly hint instead of a traceback.
- Module layout: `polaris/console/` with `banner.py` (art + schemes + renderer)
  and `repl.py` (`PolarisConsole(cmd.Cmd)` + `run()`); the CLI imports lazily so
  `polaris info` stays torch-free.

## Non-goals

- No `pyfiglet`/`rich` dependency (raw ANSI suffices; deps can come later if art
  needs grow).
- No training/evaluation commands in the console yet — inference-first; grow
  per real need (ADR-0004 spirit).

## Tests (offline)

Seeded scheme selection; NO_COLOR/non-TTY suppression; REPL commands driven via
`onecmd` against a tiny saved bundle; CLI wiring.
