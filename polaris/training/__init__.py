"""
Training subsystem for Polaris.
"""

from __future__ import annotations

from polaris.training.checkpoint import load_checkpoint, save_checkpoint
from polaris.training.config import TrainingConfig
from polaris.training.loop import train
from polaris.training.scheduler import WarmupSchedule
from polaris.training.trainer import EpochRecord, Trainer, TrainingResult

__all__ = [
    "EpochRecord",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "WarmupSchedule",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
