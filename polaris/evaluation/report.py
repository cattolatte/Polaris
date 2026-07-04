"""A structured classification report and an evaluation harness.

Composes the metric primitives from :mod:`polaris.evaluation.metrics` into a
single :class:`ClassificationReport` — per-class and averaged precision, recall,
and F1, plus support, accuracy, and the confusion matrix — with a readable text
rendering, and a harness to evaluate a model over a set of batches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.evaluation.metrics import (
    accuracy as compute_accuracy,
)
from polaris.evaluation.metrics import (
    confusion_matrix,
    precision_recall_f1,
    predict,
)

__all__ = ["ClassificationReport", "classification_report", "evaluate_model"]


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """A structured classification evaluation report.

    Holds per-class precision / recall / F1 / support, overall accuracy, the
    macro- and support-weighted averages of the three metrics, and the confusion
    matrix. It is a value object; :meth:`to_text` is one rendering of it.
    """

    class_names: tuple[str, ...]
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]
    support: tuple[int, ...]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion: tuple[tuple[int, ...], ...]

    def to_text(self) -> str:
        """Render the report as a readable table."""
        rows = [
            f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
        ]
        for index, name in enumerate(self.class_names):
            rows.append(
                f"{name:<12}{self.precision[index]:>10.4f}"
                f"{self.recall[index]:>10.4f}{self.f1[index]:>10.4f}"
                f"{self.support[index]:>10}"
            )
        total = sum(self.support)
        rows.append("")
        rows.append(
            f"{'accuracy':<12}{'':>10}{'':>10}{self.accuracy:>10.4f}{total:>10}"
        )
        rows.append(
            f"{'macro avg':<12}{self.macro_precision:>10.4f}"
            f"{self.macro_recall:>10.4f}{self.macro_f1:>10.4f}{total:>10}"
        )
        rows.append(
            f"{'weighted avg':<12}{self.weighted_precision:>10.4f}"
            f"{self.weighted_recall:>10.4f}{self.weighted_f1:>10.4f}{total:>10}"
        )

        rows.append("")
        rows.append("confusion matrix (rows = true, cols = predicted):")
        rows.append(
            "            " + "".join(f"{name:>10}" for name in self.class_names)
        )
        for index, name in enumerate(self.class_names):
            counts = "".join(f"{value:>10}" for value in self.confusion[index])
            rows.append(f"{name:<12}{counts}")
        return "\n".join(rows)


def classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> ClassificationReport:
    """Build a :class:`ClassificationReport` from logits and labels.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape ``(n, num_classes)``.
    labels : torch.Tensor
        Ground-truth labels of shape ``(n,)``.
    num_classes : int
        The number of classes (must be at least 1).
    class_names : Sequence[str], optional
        Names for each class; defaults to the string class indices. Must have
        length ``num_classes`` if given.

    Raises
    ------
    ValueError
        If ``num_classes`` is less than 1, or ``class_names`` has the wrong
        length.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}")
    if class_names is None:
        names = tuple(str(index) for index in range(num_classes))
    else:
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries, "
                f"expected {num_classes}"
            )
        names = tuple(class_names)

    precisions, recalls, f1s = precision_recall_f1(
        logits, labels, num_classes=num_classes
    )
    matrix = confusion_matrix(logits, labels, num_classes=num_classes)
    support = [int(matrix[cls, :].sum().item()) for cls in range(num_classes)]
    total = sum(support)

    def _weighted(values: list[float]) -> float:
        if total == 0:
            return 0.0
        return (
            sum(value * count for value, count in zip(values, support, strict=True))
            / total
        )

    confusion = tuple(
        tuple(int(value) for value in matrix[cls].tolist())
        for cls in range(num_classes)
    )

    return ClassificationReport(
        class_names=names,
        precision=tuple(precisions),
        recall=tuple(recalls),
        f1=tuple(f1s),
        support=tuple(support),
        accuracy=compute_accuracy(logits, labels),
        macro_precision=sum(precisions) / num_classes,
        macro_recall=sum(recalls) / num_classes,
        macro_f1=sum(f1s) / num_classes,
        weighted_precision=_weighted(precisions),
        weighted_recall=_weighted(recalls),
        weighted_f1=_weighted(f1s),
        confusion=confusion,
    )


def evaluate_model(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> ClassificationReport:
    """Evaluate ``model`` over ``batches`` and return a classification report.

    A thin harness decoupled from training: it runs the model with
    :func:`~polaris.evaluation.metrics.predict` and builds the report.
    """
    logits, labels = predict(model, batches)
    return classification_report(
        logits, labels, num_classes=num_classes, class_names=class_names
    )
