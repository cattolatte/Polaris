"""Unit tests for the contrastive-training driver."""

from __future__ import annotations

import random

import pytest
import torch

from polaris.collation import collate_contrastive
from polaris.models import TextEmbedder
from polaris.tokenizers.encoding import Encoding
from polaris.training import train_contrastive

VOCAB_SIZE = 40


def _enc(*ids: int) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(f"t{i}" for i in ids))


def _embedder() -> TextEmbedder:
    torch.manual_seed(0)
    return TextEmbedder(
        vocab_size=VOCAB_SIZE, embed_dim=32, num_heads=4, num_layers=2, max_len=16
    )


def _pair_batches() -> list:
    random.seed(0)

    def review() -> Encoding:
        return _enc(*[random.randint(3, VOCAB_SIZE - 1) for _ in range(5)])

    return [
        collate_contrastive(
            [(review(), review()) for _ in range(8)], pad_id=0, max_length=16
        )
        for _ in range(3)
    ]


# --- learning behavior ---


def test_training_reduces_loss() -> None:
    """The InfoNCE loss trends down over epochs."""
    embedder = _embedder()
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)

    history = train_contrastive(
        embedder, _pair_batches(), optimizer=optimizer, epochs=30, seed=0
    )

    assert len(history) == 30
    assert history[-1] < history[0]


def test_hard_negative_batches_train() -> None:
    """A batch carrying hard negatives runs end to end and returns losses."""
    embedder = _embedder()
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)
    batch = collate_contrastive(
        [
            (_enc(3, 4), _enc(5, 6), [_enc(7, 8), _enc(9, 10)]),
            (_enc(11, 12), _enc(13, 14), [_enc(15, 16), _enc(17, 18)]),
        ],
        pad_id=0,
        max_length=16,
    )

    history = train_contrastive(
        embedder, [batch], optimizer=optimizer, epochs=3, seed=0
    )

    assert len(history) == 3
    assert all(loss >= 0.0 for loss in history)


# --- validation ---


def test_empty_batches_raise() -> None:
    """Training with no batches is a usage error."""
    embedder = _embedder()
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="empty"):
        train_contrastive(embedder, [], optimizer=optimizer, epochs=1)


def test_non_positive_epochs_raise() -> None:
    """A non-positive epoch count is a usage error."""
    embedder = _embedder()
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="epochs must be"):
        train_contrastive(embedder, _pair_batches(), optimizer=optimizer, epochs=0)
