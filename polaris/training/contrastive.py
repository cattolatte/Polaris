"""A minimal contrastive-training driver for the text embedder.

Trains a :class:`~polaris.models.embedder.TextEmbedder` with the InfoNCE objective
(:func:`~polaris.training.losses.info_nce_loss`). Like
:func:`~polaris.training.loop.train` and
:func:`~polaris.pretraining.loop.pretrain`, it is a small concrete loop rather than
a generalized abstraction — the embedder is scored on *pairs*, not per-row labels,
so it doesn't fit the classifier trainer (ADR-0004).

The two-stage curriculum a retriever needs (e.g. general triples then in-domain
pairs) is the *caller's* concern — this driver just consumes whatever
:class:`~polaris.collation.contrastive.ContrastiveBatch` batches it is given.

Design Principles
-----------------
- Reuses the embedder's own forward (encoder -> mean-pool -> normalize); no model
  math here.
- Seeded and offline; returns the per-epoch mean loss, mirroring ``train``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from polaris.collation.contrastive import ContrastiveBatch
from polaris.models.embedder import TextEmbedder
from polaris.training.losses import info_nce_loss
from polaris.utils.device import module_device

__all__ = ["train_contrastive"]


def train_contrastive(
    embedder: TextEmbedder,
    batches: Sequence[ContrastiveBatch],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    temperature: float = 0.05,
    symmetric: bool = True,
    seed: int = 0,
) -> list[float]:
    """Train ``embedder`` on contrastive batches with the InfoNCE objective.

    Parameters
    ----------
    embedder : TextEmbedder
        The bi-encoder tower to train (it should L2-normalize its output).
    batches : Sequence[ContrastiveBatch]
        Anchor/positive (and optional hard-negative) batches, iterated per epoch.
    optimizer : torch.optim.Optimizer
        The optimizer updating the embedder's parameters.
    epochs : int
        Number of passes over ``batches``. Must be at least ``1``.
    temperature : float, default 0.05
        InfoNCE temperature.
    symmetric : bool, default True
        Whether to average both contrastive directions (see :func:`info_nce_loss`).
    seed : int, default 0
        Seed set once for reproducibility.

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

    torch.manual_seed(seed)
    device = module_device(embedder)
    batches = [batch.to(device) for batch in batches]

    embedder.train()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        running_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            anchor_emb = embedder(batch.anchor)  # (B, D)
            positive_emb = embedder(batch.positive)  # (B, D)

            extra_negatives: torch.Tensor | None = None
            if batch.negatives is not None:
                negative_emb = embedder(batch.negatives)  # (B * N, D)
                extra_negatives = negative_emb.view(
                    len(batch), batch.num_negatives, -1
                )  # (B, N, D)

            loss = info_nce_loss(
                anchor_emb,
                positive_emb,
                temperature=temperature,
                extra_negatives=extra_negatives,
                symmetric=symmetric,
            )
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub is untyped
            optimizer.step()
            running_loss += float(loss.item())
        epoch_losses.append(running_loss / len(batches))

    return epoch_losses
