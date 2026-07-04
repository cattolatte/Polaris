"""Unit tests for :class:`polaris.training.config.TrainingConfig`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from polaris.training.config import TrainingConfig


def test_defaults() -> None:
    """The config exposes sensible defaults."""
    config = TrainingConfig()

    assert config.epochs == 5
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.warmup_ratio == pytest.approx(0.1)
    assert config.early_stopping_patience is None
    assert config.checkpoint_path is None


def test_accepts_overrides() -> None:
    """Fields can be overridden."""
    config = TrainingConfig(epochs=10, learning_rate=5e-4, early_stopping_patience=3)

    assert config.epochs == 10
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.early_stopping_patience == 3


def test_zero_epochs_is_rejected() -> None:
    """Epochs must be at least one."""
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)


def test_non_positive_learning_rate_is_rejected() -> None:
    """The learning rate must be positive."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=0.0)


def test_out_of_range_warmup_ratio_is_rejected() -> None:
    """The warmup ratio must lie in [0, 1]."""
    with pytest.raises(ValidationError):
        TrainingConfig(warmup_ratio=1.5)


def test_round_trips_to_and_from_a_file(tmp_path: Path) -> None:
    """A config snapshot written to disk loads back equal."""
    config = TrainingConfig(epochs=7, learning_rate=5e-4, early_stopping_patience=2)
    path = tmp_path / "config.json"

    config.to_file(path)
    loaded = TrainingConfig.from_file(path)

    assert loaded == config
