"""
Training subsystem for Polaris.
"""

from __future__ import annotations

from polaris.training.checkpoint import load_checkpoint, save_checkpoint
from polaris.training.config import TrainingConfig
from polaris.training.contrastive import train_contrastive
from polaris.training.loop import train
from polaris.training.losses import info_nce_loss
from polaris.training.scheduler import WarmupSchedule
from polaris.training.trainer import EpochRecord, Trainer, TrainingResult

__all__ = [
    "EpochRecord",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "WarmupSchedule",
    "info_nce_loss",
    "load_checkpoint",
    "save_checkpoint",
    "train",
    "train_contrastive",
]
