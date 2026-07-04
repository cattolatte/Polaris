"""Unit tests for the transformer-encoder classifier.

The learning-behaviour test trains on a tiny offline synthetic dataset through
the real training loop, asserting the loss decreases — no dataset is downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from polaris.collation.collator import collate
from polaris.models.transformer_classifier import TransformerEncoderClassifier
from polaris.tokenizers.encoding import Encoding
from polaris.training.loop import train
from polaris.utils.seed import set_seed


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


def _model() -> TransformerEncoderClassifier:
    return TransformerEncoderClassifier(
        vocab_size=3,
        num_classes=2,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=16,
        dropout=0.0,
        pad_id=0,
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_returns_logits_of_expected_shape() -> None:
    """Logits have shape (batch_size, num_classes)."""
    model = TransformerEncoderClassifier(
        vocab_size=10, num_classes=3, embed_dim=16, num_heads=2, num_layers=2
    )
    batch = collate([(_encoding([1, 2, 3]), 0), (_encoding([4]), 1)], pad_id=0)

    assert model(batch).shape == (2, 3)


# ---------------------------------------------------------------------------
# Learning behaviour
# ---------------------------------------------------------------------------


def test_transformer_reduces_loss_on_separable_data() -> None:
    """The transformer trains through the v0.4 loop and lowers the loss."""
    set_seed(0)
    samples = [
        (_encoding([1, 1]), 0),
        (_encoding([2, 2]), 1),
        (_encoding([1]), 0),
        (_encoding([2]), 1),
    ]
    batch = collate(samples, pad_id=0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = train(model, [batch], optimizer=optimizer, epochs=50)

    assert losses[-1] < losses[0]
