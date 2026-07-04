"""Basic evaluation metrics.

The minimal metrics needed to know whether training worked: accuracy over a set
of predictions, and a convenience ``evaluate`` that runs a model over batches
without training. This is the seed of the fuller evaluation framework planned for
v0.7 (precision/recall/F1, confusion matrices, benchmark reports).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.utils.device import module_device

__all__ = [
    "accuracy",
    "confusion_matrix",
    "evaluate",
    "precision_recall_f1",
    "predict",
]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape ``(n, num_classes)``. The prediction is the argmax.
    labels : torch.Tensor
        Ground-truth labels of shape ``(n,)``.

    Returns
    -------
    float
        The fraction of correct predictions, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``logits`` and ``labels`` disagree on ``n``, or ``n`` is ``0``.
    """
    if logits.shape[0] != labels.shape[0]:
        msg = (
            "logits and labels must have the same number of rows, got "
            f"{logits.shape[0]} and {labels.shape[0]}"
        )
        raise ValueError(msg)
    if labels.shape[0] == 0:
        msg = "cannot compute accuracy over zero predictions"
        raise ValueError(msg)

    predictions = logits.argmax(dim=-1)
    correct = int((predictions == labels).sum().item())
    return correct / int(labels.shape[0])


def evaluate(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    loss_fn: nn.Module | None = None,
) -> tuple[float, float]:
    """Evaluate ``model`` over ``batches`` without training.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate. Called with a :class:`Batch`, returns logits.
    batches : Sequence[Batch]
        The evaluation batches.
    loss_fn : nn.Module, optional
        The loss criterion. Defaults to :class:`torch.nn.CrossEntropyLoss`.

    Returns
    -------
    tuple[float, float]
        The mean loss per batch and the overall accuracy across all examples.

    Raises
    ------
    ValueError
        If ``batches`` is empty.
    """
    if not batches:
        msg = "cannot evaluate on an empty sequence of batches"
        raise ValueError(msg)

    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    device = module_device(model)
    batches = [batch.to(device) for batch in batches]

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in batches:
            logits: torch.Tensor = model(batch)
            loss: torch.Tensor = criterion(logits, batch.labels)
            total_loss += float(loss.item())
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions == batch.labels).sum().item())
            total_examples += len(batch)

    return total_loss / len(batches), total_correct / total_examples


def predict(
    model: nn.Module,
    batches: Sequence[Batch],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``batches``, returning concatenated logits and labels.

    Parameters
    ----------
    model : nn.Module
        The model to run. Called with a :class:`Batch`, returns logits.
    batches : Sequence[Batch]
        The batches to run over.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The logits of shape ``(n, num_classes)`` and labels of shape ``(n,)``
        gathered across all batches.

    Raises
    ------
    ValueError
        If ``batches`` is empty.
    """
    if not batches:
        msg = "cannot predict over an empty sequence of batches"
        raise ValueError(msg)

    device = module_device(model)
    model.eval()
    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            moved = batch.to(device)
            logit_chunks.append(model(moved))
            label_chunks.append(moved.labels)
    return torch.cat(logit_chunks), torch.cat(label_chunks)


def confusion_matrix(
    logits: torch.Tensor, labels: torch.Tensor, *, num_classes: int
) -> torch.Tensor:
    """Return the confusion matrix (rows = true class, columns = predicted).

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape ``(n, num_classes)``. The prediction is the argmax.
    labels : torch.Tensor
        Ground-truth labels of shape ``(n,)``.
    num_classes : int
        The number of classes.

    Returns
    -------
    torch.Tensor
        A ``(num_classes, num_classes)`` long tensor of counts.
    """
    predictions = logits.argmax(dim=-1)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, predicted in zip(
        labels.tolist(), predictions.tolist(), strict=True
    ):
        matrix[true_label, predicted] += 1
    return matrix


def precision_recall_f1(
    logits: torch.Tensor, labels: torch.Tensor, *, num_classes: int
) -> tuple[list[float], list[float], list[float]]:
    """Return per-class precision, recall, and F1.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape ``(n, num_classes)``.
    labels : torch.Tensor
        Ground-truth labels of shape ``(n,)``.
    num_classes : int
        The number of classes.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Per-class precision, recall, and F1 (each a list of length
        ``num_classes``). A metric is ``0.0`` where its denominator is zero.
    """
    matrix = confusion_matrix(logits, labels, num_classes=num_classes)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(num_classes):
        true_positive = int(matrix[cls, cls].item())
        predicted_positive = int(matrix[:, cls].sum().item())
        actual_positive = int(matrix[cls, :].sum().item())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s
