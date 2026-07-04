"""Unit tests for :mod:`polaris.evaluation.metrics`.

All tests run fully offline on tiny tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from polaris.collation.collator import collate
from polaris.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    evaluate,
    precision_recall_f1,
    predict,
)
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


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_concatenates_logits_and_labels() -> None:
    """``predict`` gathers logits and labels across batches, in order."""
    model = MeanPoolingClassifier(
        vocab_size=5, num_classes=2, embedding_dim=8, pad_id=0
    )
    first = collate([(_encoding([1, 2]), 0)], pad_id=0)
    second = collate([(_encoding([3]), 1), (_encoding([4]), 0)], pad_id=0)

    logits, labels = predict(model, [first, second])

    assert logits.shape == (3, 2)
    assert labels.tolist() == [0, 1, 0]


def test_predict_empty_raises() -> None:
    """Predicting over no batches is rejected."""
    model = MeanPoolingClassifier(vocab_size=5, num_classes=2)

    with pytest.raises(ValueError, match="empty"):
        predict(model, [])


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_counts_true_vs_predicted() -> None:
    """Rows are true classes; columns are predicted classes."""
    # argmax -> [0, 0, 1, 1]; labels -> [0, 1, 1, 1]
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.0], [1.0, 2.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 1])

    matrix = confusion_matrix(logits, labels, num_classes=2)

    assert matrix.tolist() == [[1, 0], [1, 2]]


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------


def test_precision_recall_f1_per_class() -> None:
    """Per-class precision, recall, and F1 match a hand-computed case."""
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.0], [1.0, 2.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 1])

    precisions, recalls, f1s = precision_recall_f1(logits, labels, num_classes=2)

    assert precisions[0] == pytest.approx(0.5)
    assert recalls[0] == pytest.approx(1.0)
    assert precisions[1] == pytest.approx(1.0)
    assert recalls[1] == pytest.approx(2 / 3)
    assert f1s[1] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
