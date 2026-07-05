# Experiments Module

Make runs **reproducible and comparable**: capture the environment and record each
run's configuration, metric history, and evaluation report to disk. This is what
turns "I got 85%" into a recorded, re-runnable result (the benchmarks in
`BENCHMARKS.md` are these records).

## Public surface

- `capture_environment()` (`environment.py`) — a snapshot of the run environment
  (Polaris, Python, platform, and PyTorch versions).
- `record_run(directory, *, config, history, report, environment, seed)`
  (`tracker.py`) — write a run to `directory/` as `config.json`, `metrics.json`,
  `report.txt`, `environment.json`, and `run.json`.
- `load_run(directory) -> RunData` / `RunData` (`tracker.py`) — read a recorded run
  back as a structured value object.

## Design note

A single, concrete file-based tracker — no pluggable backends or experiment-tracking
framework until a second real consumer appears (ADR-0004/0005).
