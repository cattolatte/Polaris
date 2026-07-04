"""A minimal training loop.

Deliberately a plain function, not a ``Trainer`` class: only one concrete
training use exists so far, so no abstraction is justified yet (see
``docs/adr/0004-abstraction-policy.md``). The loop is the textbook one: for each
epoch, for each batch, run a forward pass, compute the loss, backpropagate, and
step the optimizer.

Checkpointing, callbacks, schedulers, and configuration are intentionally absent
— they arrive with the mature training engine in v0.6.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.utils.device import module_device

__all__ = ["train"]


def train(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    loss_fn: nn.Module | None = None,
) -> list[float]:
    """Train ``model`` on ``batches`` for a number of epochs.

    Parameters
    ----------
    model : nn.Module
        The model to train. It is called with a :class:`Batch` and must return
        logits of shape ``(batch_size, num_classes)``.
    batches : Sequence[Batch]
        The training batches, iterated once per epoch.
    optimizer : torch.optim.Optimizer
        The optimizer updating ``model``'s parameters.
    epochs : int
        Number of passes over ``batches``. Must be at least ``1``.
    loss_fn : nn.Module, optional
        The loss criterion. Defaults to :class:`torch.nn.CrossEntropyLoss`.

    Returns
    -------
    list[float]
        The mean loss for each epoch (length ``epochs``).

    Raises
    ------
    ValueError
        If ``epochs`` is less than ``1`` or ``batches`` is empty.
    """
    if epochs < 1:
        msg = f"epochs must be at least 1, got {epochs}"
        raise ValueError(msg)
    if not batches:
        msg = "cannot train on an empty sequence of batches"
        raise ValueError(msg)

    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    # Move the batches to wherever the model lives (CPU / MPS / CUDA), once.
    device = module_device(model)
    batches = [batch.to(device) for batch in batches]

    model.train()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        running_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            logits: torch.Tensor = model(batch)
            loss: torch.Tensor = criterion(logits, batch.labels)
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub is untyped
            optimizer.step()
            running_loss += float(loss.item())
        epoch_losses.append(running_loss / len(batches))

    return epoch_losses
