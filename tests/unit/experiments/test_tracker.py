"""Unit tests for :mod:`polaris.experiments.tracker`.

Runs are written to a pytest temp directory — offline, no network.
"""

from __future__ import annotations

from pathlib import Path

import torch

from polaris.evaluation.report import ClassificationReport, classification_report
from polaris.experiments.tracker import RunData, load_run, record_run
from polaris.training.config import TrainingConfig
from polaris.training.trainer import EpochRecord


def _history() -> tuple[EpochRecord, ...]:
    return (
        EpochRecord(epoch=1, train_loss=0.5, val_loss=0.6, val_accuracy=0.7),
        EpochRecord(epoch=2, train_loss=0.4, val_loss=0.55, val_accuracy=0.75),
    )


def _report() -> ClassificationReport:
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1])
    return classification_report(logits, labels, num_classes=2)


def test_record_writes_the_expected_files(tmp_path: Path) -> None:
    """Recording a run writes all the artifact files."""
    run_dir = record_run(
        tmp_path / "run",
        config=TrainingConfig(epochs=3),
        history=_history(),
        report=_report(),
        environment={"polaris": "0.8.0", "python": "3.12"},
        seed=42,
    )

    for name in ("config.json", "metrics.json", "report.txt", "run.json"):
        assert (run_dir / name).exists()


def test_run_round_trips(tmp_path: Path) -> None:
    """A recorded run loads back with its config, history, environment, and seed."""
    config = TrainingConfig(epochs=3, learning_rate=5e-4)
    history = _history()
    environment = {"polaris": "0.8.0", "python": "3.12"}

    record_run(
        tmp_path / "run",
        config=config,
        history=history,
        report=_report(),
        environment=environment,
        seed=42,
    )
    loaded = load_run(tmp_path / "run")

    assert isinstance(loaded, RunData)
    assert loaded.config == config
    assert loaded.history == history
    assert loaded.environment == environment
    assert loaded.seed == 42
