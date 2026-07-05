"""Unit tests for the masked-language-model pretraining loop."""

from __future__ import annotations

import math
import random

import pytest
import torch

from polaris.collation.batch import Batch
from polaris.pretraining.loop import pretrain
from polaris.pretraining.model import MaskedLanguageModel

VOCAB_SIZE = 12


def _batch(rows: list[list[int]]) -> Batch:
    ids = torch.tensor(rows)
    return Batch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=torch.zeros(len(rows), dtype=torch.long),
    )


def _fixture_batches() -> list[Batch]:
    random.seed(0)
    rows = [[random.randint(3, VOCAB_SIZE - 1) for _ in range(20)] for _ in range(24)]
    return [_batch(rows[i : i + 8]) for i in range(0, 24, 8)]


def _model() -> MaskedLanguageModel:
    torch.manual_seed(0)
    return MaskedLanguageModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        max_len=32,
        pad_id=0,
    )


# --- learning behavior ---


def test_pretraining_reduces_loss() -> None:
    """Masked-token loss trends down over epochs on a learnable fixture."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    history = pretrain(
        model,
        optimizer,
        _fixture_batches(),
        mask_id=2,
        vocab_size=VOCAB_SIZE,
        special_token_ids=(0, 1, 2),
        epochs=60,
        seed=0,
    )

    losses = [record.loss for record in history]
    assert not any(math.isnan(loss) for loss in losses)
    assert sum(losses[-5:]) / 5 < losses[0]


def test_pretraining_records_one_entry_per_epoch() -> None:
    """The returned history has exactly ``epochs`` records, numbered from 1."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    history = pretrain(
        model,
        optimizer,
        _fixture_batches(),
        mask_id=2,
        vocab_size=VOCAB_SIZE,
        special_token_ids=(0, 1, 2),
        epochs=3,
        seed=0,
    )

    assert [record.epoch for record in history] == [1, 2, 3]


# --- validation ---


def test_empty_batches_raise() -> None:
    """Pretraining with no batches is a usage error."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="non-empty"):
        pretrain(model, optimizer, [], mask_id=2, vocab_size=VOCAB_SIZE)


def test_non_positive_epochs_raise() -> None:
    """A non-positive epoch count is a usage error."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="epochs must be"):
        pretrain(
            model,
            optimizer,
            _fixture_batches(),
            mask_id=2,
            vocab_size=VOCAB_SIZE,
            epochs=0,
        )
