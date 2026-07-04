"""Unit tests for :func:`polaris.training.loop.train`.

These assert *learning behaviour* on a tiny, offline synthetic dataset — not
exact float values. No real dataset is downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from polaris.collation.batch import Batch
from polaris.collation.collator import collate
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.tokenizers.encoding import Encoding
from polaris.training.loop import train
from polaris.utils.seed import set_seed


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


def _separable_batch() -> Batch:
    # Token 1 => class 0, token 2 => class 1. Trivially separable.
    samples = [
        (_encoding([1, 1]), 0),
        (_encoding([2, 2]), 1),
        (_encoding([1]), 0),
        (_encoding([2]), 1),
    ]
    return collate(samples, pad_id=0)


def _fresh_model() -> MeanPoolingClassifier:
    return MeanPoolingClassifier(vocab_size=3, num_classes=2, embedding_dim=8, pad_id=0)


# ---------------------------------------------------------------------------
# Learning behaviour
# ---------------------------------------------------------------------------


def test_training_reduces_loss_on_separable_data() -> None:
    """Loss at the end of training is lower than at the start."""
    set_seed(0)
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    losses = train(model, [_separable_batch()], optimizer=optimizer, epochs=50)

    assert losses[-1] < losses[0]


def test_returns_one_loss_per_epoch() -> None:
    """The loop returns exactly one mean loss per epoch."""
    set_seed(0)
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = train(model, [_separable_batch()], optimizer=optimizer, epochs=7)

    assert len(losses) == 7


def test_training_is_reproducible_with_seed() -> None:
    """Seeding before setup makes a whole training run reproducible."""

    def run() -> list[float]:
        set_seed(0)
        model = _fresh_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        return train(model, [_separable_batch()], optimizer=optimizer, epochs=5)

    assert run() == run()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_epochs_raises() -> None:
    """Fewer than one epoch is rejected."""
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="epochs"):
        train(model, [_separable_batch()], optimizer=optimizer, epochs=0)


def test_empty_batches_raises() -> None:
    """Training on no batches is rejected."""
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="empty"):
        train(model, [], optimizer=optimizer, epochs=1)
