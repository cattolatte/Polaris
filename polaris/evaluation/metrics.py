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

__all__ = ["accuracy", "evaluate"]


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
