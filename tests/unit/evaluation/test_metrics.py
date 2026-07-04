"""Unit tests for :mod:`polaris.evaluation.metrics`.

All tests run fully offline on tiny tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from polaris.collation.collator import collate
from polaris.evaluation.metrics import accuracy, evaluate
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.tokenizers.encoding import Encoding


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_all_correct_is_one() -> None:
    """Perfect predictions score 1.0."""
    logits = torch.tensor([[2.0, 1.0], [0.0, 1.0]])  # argmax -> [0, 1]
    labels = torch.tensor([0, 1])

    assert accuracy(logits, labels) == 1.0


def test_half_correct_is_one_half() -> None:
    """Half-correct predictions score 0.5."""
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.0]])  # argmax -> [0, 0]
    labels = torch.tensor([0, 1])

    assert accuracy(logits, labels) == 0.5


def test_all_wrong_is_zero() -> None:
    """No correct predictions score 0.0."""
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.0]])  # argmax -> [0, 0]
    labels = torch.tensor([1, 1])

    assert accuracy(logits, labels) == 0.0


def test_accuracy_shape_mismatch_raises() -> None:
    """Mismatched logits and labels are rejected."""
    with pytest.raises(ValueError, match="same number of rows"):
        accuracy(torch.zeros((2, 3)), torch.zeros((3,), dtype=torch.long))


def test_accuracy_empty_raises() -> None:
    """Accuracy over zero predictions is rejected."""
    with pytest.raises(ValueError, match="zero predictions"):
        accuracy(torch.zeros((0, 3)), torch.zeros((0,), dtype=torch.long))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_loss_and_accuracy_in_range() -> None:
    """``evaluate`` returns a finite loss and an accuracy in [0, 1]."""
    model = MeanPoolingClassifier(
        vocab_size=5, num_classes=2, embedding_dim=8, pad_id=0
    )
    batch = collate([(_encoding([1, 2]), 0), (_encoding([3]), 1)], pad_id=0)

    loss, acc = evaluate(model, [batch])

    assert loss > 0.0
    assert 0.0 <= acc <= 1.0


def test_evaluate_empty_raises() -> None:
    """Evaluating on no batches is rejected."""
    model = MeanPoolingClassifier(
        vocab_size=5, num_classes=2, embedding_dim=8, pad_id=0
    )

    with pytest.raises(ValueError, match="empty"):
        evaluate(model, [])
