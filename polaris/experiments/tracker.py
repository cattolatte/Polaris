"""A local experiment tracker.

Records a run — its configuration, metric history, classification report, and
environment — to a directory on disk, and reads the structured parts back. This
is the single, concrete backend (local filesystem); there is no tracking
abstraction until a second backend earns one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from polaris.evaluation.report import ClassificationReport
from polaris.training.config import TrainingConfig
from polaris.training.trainer import EpochRecord

__all__ = ["RunData", "record_run", "load_run"]


@dataclass(frozen=True, slots=True)
class RunData:
    """The structured contents of a recorded run."""

    config: TrainingConfig
    history: tuple[EpochRecord, ...]
    environment: dict[str, str]
    seed: int


def record_run(
    directory: str | Path,
    *,
    config: TrainingConfig,
    history: Sequence[EpochRecord],
    report: ClassificationReport,
    environment: dict[str, str],
    seed: int,
) -> Path:
    """Write a run to ``directory`` and return its path.

    Writes ``config.json``, ``metrics.json``, ``report.txt``,
    ``environment.json``, and ``run.json`` (seed and headline accuracy).
    """
    run_dir = Path(directory)
    run_dir.mkdir(parents=True, exist_ok=True)

    config.to_file(run_dir / "config.json")
    (run_dir / "metrics.json").write_text(
        json.dumps([asdict(record) for record in history], indent=2)
    )
    (run_dir / "report.txt").write_text(report.to_text())
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (run_dir / "run.json").write_text(
        json.dumps({"seed": seed, "accuracy": report.accuracy}, indent=2)
    )
    return run_dir


def load_run(directory: str | Path) -> RunData:
    """Read a run recorded by :func:`record_run` back into a :class:`RunData`."""
    run_dir = Path(directory)

    config = TrainingConfig.from_file(run_dir / "config.json")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    run_meta = json.loads((run_dir / "run.json").read_text())

    history = tuple(EpochRecord(**record) for record in metrics)
    return RunData(
        config=config,
        history=history,
        environment=environment,
        seed=int(run_meta["seed"]),
    )
