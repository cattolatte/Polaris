"""Typed training configuration.

A run is described by a ``TrainingConfig`` rather than edited constants, so
experiments are reproducible and validated. This is where Polaris' configuration
system first appears — introduced because the training engine genuinely needs it
(see the Phase 06 design doc), not speculatively.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["TrainingConfig"]


class TrainingConfig(BaseModel):
    """Hyperparameters for a training run.

    Parameters
    ----------
    epochs : int
        Number of passes over the training data.
    learning_rate : float
        The peak learning rate (reached after warmup).
    warmup_ratio : float
        Fraction of total steps spent warming the learning rate up, in ``[0, 1]``.
    weight_decay : float
        L2 weight decay for the optimizer.
    early_stopping_patience : int or None
        Stop after this many epochs without validation improvement. ``None``
        disables early stopping (requires validation data to take effect).
    checkpoint_path : str or None
        If set, the best model (by validation accuracy) is written here.
    """

    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    checkpoint_path: str | None = None

    def to_file(self, path: str | Path) -> None:
        """Write the config to a JSON file (a config snapshot)."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingConfig:
        """Load a config from a JSON file written by :meth:`to_file`."""
        return cls.model_validate_json(Path(path).read_text())
