"""The Polaris training engine.

``Trainer`` composes the pieces a real training run needs — a warmup learning-rate
schedule, a validation loop, early stopping, best-model checkpointing, and
logging — around a model and optimizer, driven by a :class:`TrainingConfig`.

It is intentionally the *vehicle* for these features (they need cross-epoch
state); the minimal :func:`~polaris.training.loop.train` function remains for the
simple case. There is deliberately no generic callback system — early stopping
and logging are built in — until a real extension need appears.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.evaluation.metrics import evaluate
from polaris.training.checkpoint import save_checkpoint
from polaris.training.config import TrainingConfig
from polaris.training.scheduler import WarmupSchedule
from polaris.utils.device import module_device
from polaris.utils.logging import get_logger

__all__ = ["Trainer", "TrainingResult", "EpochRecord"]


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Metrics recorded for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_accuracy: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The outcome of a training run."""

    history: tuple[EpochRecord, ...]
    best_val_accuracy: float | None


class Trainer:
    """Train a model with scheduling, validation, early stopping, and checkpoints.

    Parameters
    ----------
    model : nn.Module
        The model to train (called with a :class:`Batch`, returns logits).
    optimizer : torch.optim.Optimizer
        The optimizer.
    config : TrainingConfig
        The run configuration.
    loss_fn : nn.Module, optional
        Loss criterion. Defaults to :class:`torch.nn.CrossEntropyLoss`.
    logger : logging.Logger, optional
        Logger for per-epoch metrics. Defaults to the Polaris logger.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        *,
        loss_fn: nn.Module | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.logger = logger if logger is not None else get_logger()

    def fit(
        self,
        train_batches: Sequence[Batch],
        val_batches: Sequence[Batch] | None = None,
    ) -> TrainingResult:
        """Train the model and return its history and best validation accuracy.

        Raises
        ------
        ValueError
            If ``train_batches`` is empty.
        """
        if not train_batches:
            msg = "cannot train on an empty sequence of batches"
            raise ValueError(msg)

        device = module_device(self.model)
        train_data = [batch.to(device) for batch in train_batches]
        val_data = (
            [batch.to(device) for batch in val_batches]
            if val_batches is not None
            else None
        )

        total_steps = self.config.epochs * len(train_data)
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        schedule = WarmupSchedule(
            self.optimizer,
            base_lr=self.config.learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        history: list[EpochRecord] = []
        best_val_accuracy: float | None = None
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_data, schedule)

            val_loss: float | None = None
            val_accuracy: float | None = None
            if val_data is not None:
                val_loss, val_accuracy = evaluate(
                    self.model, val_data, loss_fn=self.loss_fn
                )
                if best_val_accuracy is None or val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    if self.config.checkpoint_path is not None:
                        save_checkpoint(
                            self.config.checkpoint_path,
                            model=self.model,
                            optimizer=self.optimizer,
                            metadata={"epoch": epoch, "val_accuracy": val_accuracy},
                        )
                else:
                    epochs_without_improvement += 1

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
            history.append(record)
            self._log(record)

            patience = self.config.early_stopping_patience
            if (
                val_data is not None
                and patience is not None
                and epochs_without_improvement >= patience
            ):
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return TrainingResult(
            history=tuple(history), best_val_accuracy=best_val_accuracy
        )

    def _train_one_epoch(
        self, batches: Sequence[Batch], schedule: WarmupSchedule
    ) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in batches:
            self.optimizer.zero_grad()
            logits: torch.Tensor = self.model(batch)
            loss: torch.Tensor = self.loss_fn(logits, batch.labels)
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub is untyped
            self.optimizer.step()
            schedule.step()
            running_loss += float(loss.item())
        return running_loss / len(batches)

    def _log(self, record: EpochRecord) -> None:
        if record.val_accuracy is not None:
            self.logger.info(
                "epoch %d | train_loss %.4f | val_loss %.4f | val_acc %.4f",
                record.epoch,
                record.train_loss,
                record.val_loss,
                record.val_accuracy,
            )
        else:
            self.logger.info(
                "epoch %d | train_loss %.4f", record.epoch, record.train_loss
            )
