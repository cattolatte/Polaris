"""Unit tests for :class:`polaris.training.trainer.Trainer`.

Trains on a tiny offline synthetic dataset; asserts learning behaviour, early
stopping, and checkpointing — no dataset is downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from polaris.collation.batch import Batch
from polaris.collation.collator import collate
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.tokenizers.encoding import Encoding
from polaris.training.config import TrainingConfig
from polaris.training.trainer import Trainer
from polaris.utils.seed import set_seed


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


def _batch() -> Batch:
    samples = [
        (_encoding([1, 1]), 0),
        (_encoding([2, 2]), 1),
        (_encoding([1]), 0),
        (_encoding([2]), 1),
    ]
    return collate(samples, pad_id=0)


def _model() -> MeanPoolingClassifier:
    return MeanPoolingClassifier(vocab_size=3, num_classes=2, embedding_dim=8, pad_id=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_fit_reduces_train_loss() -> None:
    """The trainer lowers the training loss and records one entry per epoch."""
    set_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(model, optimizer, TrainingConfig(epochs=20, learning_rate=1e-2))

    result = trainer.fit([_batch()])

    assert len(result.history) == 20
    assert result.history[-1].train_loss < result.history[0].train_loss
    assert result.best_val_accuracy is None


def test_fit_tracks_validation_and_best_accuracy() -> None:
    """With validation data, each epoch records accuracy and the best is returned."""
    set_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(model, optimizer, TrainingConfig(epochs=15, learning_rate=1e-2))

    result = trainer.fit([_batch()], val_batches=[_batch()])

    assert result.best_val_accuracy is not None
    assert 0.0 <= result.best_val_accuracy <= 1.0
    assert all(record.val_accuracy is not None for record in result.history)


# ---------------------------------------------------------------------------
# Early stopping and checkpointing
# ---------------------------------------------------------------------------


def test_early_stopping_halts_before_max_epochs() -> None:
    """Once validation accuracy plateaus, early stopping ends training early."""
    set_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    config = TrainingConfig(epochs=50, learning_rate=5e-2, early_stopping_patience=3)
    trainer = Trainer(model, optimizer, config)

    result = trainer.fit([_batch()], val_batches=[_batch()])

    assert len(result.history) < 50


def test_best_checkpoint_is_written(tmp_path: Path) -> None:
    """A checkpoint path causes the best model to be saved."""
    set_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    checkpoint = tmp_path / "best.pt"
    config = TrainingConfig(
        epochs=10, learning_rate=5e-2, checkpoint_path=str(checkpoint)
    )

    Trainer(model, optimizer, config).fit([_batch()], val_batches=[_batch()])

    assert checkpoint.exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_batches_raises() -> None:
    """Training on no batches is rejected."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(ValueError, match="empty"):
        Trainer(model, optimizer, TrainingConfig()).fit([])
