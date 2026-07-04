"""Unit tests for :mod:`polaris.evaluation.report`.

Values are checked against a hand-computed case. All tests run offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from polaris.collation.collator import collate
from polaris.evaluation.report import (
    ClassificationReport,
    classification_report,
    evaluate_model,
)
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.tokenizers.encoding import Encoding

# argmax -> [0, 0, 1, 1]; labels -> [0, 1, 1, 1]
#   confusion = [[1, 0], [1, 2]]  (rows = true, cols = pred)
#   precision = [0.5, 1.0], recall = [1.0, 2/3], support = [1, 3], accuracy = 0.75
_LOGITS = torch.tensor([[2.0, 1.0], [2.0, 1.0], [1.0, 2.0], [1.0, 2.0]])
_LABELS = torch.tensor([0, 1, 1, 1])


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------


def test_report_matches_hand_computed_case() -> None:
    """Per-class metrics, support, accuracy, and confusion match by hand."""
    report = classification_report(_LOGITS, _LABELS, num_classes=2)

    assert report.accuracy == pytest.approx(0.75)
    assert report.precision == pytest.approx((0.5, 1.0))
    assert report.recall == pytest.approx((1.0, 2 / 3))
    assert report.support == (1, 3)
    assert report.confusion == ((1, 0), (1, 2))


def test_macro_and_weighted_averages() -> None:
    """Macro is the plain mean; weighted is support-weighted."""
    report = classification_report(_LOGITS, _LABELS, num_classes=2)

    assert report.macro_precision == pytest.approx((0.5 + 1.0) / 2)
    # weighted precision = (0.5 * 1 + 1.0 * 3) / 4
    assert report.weighted_precision == pytest.approx((0.5 * 1 + 1.0 * 3) / 4)
    assert report.weighted_recall == pytest.approx((1.0 * 1 + (2 / 3) * 3) / 4)


def test_default_and_custom_class_names() -> None:
    """Class names default to indices and can be overridden."""
    default = classification_report(_LOGITS, _LABELS, num_classes=2)
    named = classification_report(
        _LOGITS, _LABELS, num_classes=2, class_names=("neg", "pos")
    )

    assert default.class_names == ("0", "1")
    assert named.class_names == ("neg", "pos")


def test_wrong_number_of_class_names_raises() -> None:
    """Class names must match the number of classes."""
    with pytest.raises(ValueError, match="class_names"):
        classification_report(
            _LOGITS, _LABELS, num_classes=2, class_names=("only-one",)
        )


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------


def test_to_text_includes_names_and_accuracy() -> None:
    """The rendered report mentions the class names and accuracy."""
    report = classification_report(
        _LOGITS, _LABELS, num_classes=2, class_names=("neg", "pos")
    )

    text = report.to_text()

    assert "neg" in text
    assert "pos" in text
    assert "accuracy" in text
    assert "0.7500" in text


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_a_report() -> None:
    """The harness runs a model over batches and returns a report."""
    model = MeanPoolingClassifier(vocab_size=5, num_classes=2, embedding_dim=8)
    batch = collate([(_encoding([1, 2]), 0), (_encoding([3]), 1)], pad_id=0)

    report = evaluate_model(model, [batch], num_classes=2, class_names=("neg", "pos"))

    assert isinstance(report, ClassificationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert report.class_names == ("neg", "pos")
