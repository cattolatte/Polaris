"""Unit tests for :class:`polaris.models.pooling_classifier.MeanPoolingClassifier`.

All tests run fully offline on tiny tensors. They assert shapes and invariants,
never exact float values.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from polaris.collation.collator import collate
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.tokenizers.encoding import Encoding


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


def _model() -> MeanPoolingClassifier:
    return MeanPoolingClassifier(
        vocab_size=10, num_classes=3, embedding_dim=8, pad_id=0
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_returns_logits_of_expected_shape() -> None:
    """Logits have shape (batch_size, num_classes)."""
    model = _model()
    batch = collate([(_encoding([1, 2]), 0), (_encoding([3]), 1)], pad_id=0)

    logits = model(batch)

    assert logits.shape == (2, 3)


def test_forward_output_is_finite() -> None:
    """The forward pass produces finite logits."""
    model = _model()
    batch = collate([(_encoding([1, 2, 3]), 0)], pad_id=0)

    logits = model(batch)

    assert bool(torch.isfinite(logits).all())


# ---------------------------------------------------------------------------
# Masked pooling invariant
# ---------------------------------------------------------------------------


def test_padding_does_not_change_the_result() -> None:
    """Extra padding must not affect the pooled logits for the same content.

    This is the key invariant of mask-aware pooling: the same real tokens,
    with or without trailing padding, produce the same output.
    """
    model = _model()

    unpadded = collate([(_encoding([1, 2]), 0)], pad_id=0)
    # Same content, but forced to width 4 by a longer sibling in the batch.
    padded_pair = collate(
        [(_encoding([1, 2]), 0), (_encoding([3, 4, 5, 6]), 1)], pad_id=0
    )

    logits_unpadded = model(unpadded)
    logits_padded_first_row = model(padded_pair)[:1]

    assert torch.allclose(logits_unpadded, logits_padded_first_row, atol=1e-6)


# ---------------------------------------------------------------------------
# Pretrained embeddings
# ---------------------------------------------------------------------------


def test_uses_pretrained_embeddings() -> None:
    """A pretrained matrix initializes the embedding weights."""
    matrix = torch.randn(10, 8)
    model = MeanPoolingClassifier(
        vocab_size=10, num_classes=2, pad_id=0, pretrained_embeddings=matrix
    )

    assert torch.allclose(model.embedding.weight, matrix)


def test_freeze_embeddings_disables_gradient() -> None:
    """``freeze_embeddings`` makes the embedding non-trainable."""
    matrix = torch.randn(10, 8)
    model = MeanPoolingClassifier(
        vocab_size=10,
        num_classes=2,
        pretrained_embeddings=matrix,
        freeze_embeddings=True,
    )

    assert not model.embedding.weight.requires_grad
