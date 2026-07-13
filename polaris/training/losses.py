"""Training objectives that don't fit the plain ``(logits, labels)`` shape.

The generic training loop (:func:`~polaris.training.loop.train`) takes an
``nn.Module`` criterion called as ``loss_fn(logits, labels)``. Contrastive learning
doesn't fit that shape — it scores *pairs* of embeddings — so it lives here as a
function operating directly on the two embedding tensors, consumed either by
:func:`~polaris.training.contrastive.train_contrastive` or a caller's own loop.

Design Principles
-----------------
- Hand-written on tensor primitives, no external objective libraries.
- Expects **L2-normalized** embeddings (as
  :class:`~polaris.models.embedder.TextEmbedder` produces): a dot product is then a
  cosine similarity.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["info_nce_loss"]


def info_nce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    *,
    temperature: float = 0.05,
    extra_negatives: torch.Tensor | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """InfoNCE contrastive loss with in-batch and optional hard negatives.

    For each anchor ``i``, its positive is ``positive_emb[i]``; the positives of
    every other row ``j != i`` in the batch act as **in-batch negatives** (the
    standard InfoNCE trick). Any ``extra_negatives`` are *additional* mined hard
    negatives appended to each anchor's denominator.

    Parameters
    ----------
    anchor_emb : torch.Tensor
        Anchor embeddings of shape ``(B, D)``, L2-normalized.
    positive_emb : torch.Tensor
        Positive embeddings of shape ``(B, D)``, aligned with ``anchor_emb``.
    temperature : float, default 0.05
        Similarities are scaled by ``1 / temperature`` before the softmax.
    extra_negatives : torch.Tensor, optional
        Per-anchor hard negatives of shape ``(B, N, D)``. Row ``i`` is compared only
        against anchor ``i``'s own negatives.
    symmetric : bool, default True
        If ``True``, average the anchor->positive and positive->anchor directions
        (hard negatives are applied to the anchor->positive direction only).

    Returns
    -------
    torch.Tensor
        A scalar loss.

    Raises
    ------
    ValueError
        If ``anchor_emb`` and ``positive_emb`` do not share a shape, or if
        ``temperature`` is not positive.
    """
    if anchor_emb.shape != positive_emb.shape:
        msg = (
            "anchor_emb and positive_emb must share a shape, got "
            f"{tuple(anchor_emb.shape)} and {tuple(positive_emb.shape)}"
        )
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    batch_size = anchor_emb.shape[0]
    targets = torch.arange(batch_size, device=anchor_emb.device)

    # anchor -> positive: [B, B] similarities; the positive for row i sits at col i.
    logits = anchor_emb @ positive_emb.t() / temperature
    if extra_negatives is not None:
        # Each anchor scored against only its own hard negatives -> [B, N].
        hard = torch.einsum("bd,bnd->bn", anchor_emb, extra_negatives) / temperature
        logits = torch.cat([logits, hard], dim=1)  # [B, B + N]; target index unchanged
    loss = F.cross_entropy(logits, targets)

    if symmetric:
        reverse = positive_emb @ anchor_emb.t() / temperature  # [B, B]
        loss = 0.5 * (loss + F.cross_entropy(reverse, targets))

    return loss
