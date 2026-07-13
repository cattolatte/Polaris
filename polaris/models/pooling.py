"""Mask-aware pooling over token hidden states.

The mean-pool logic was written inline in
:class:`~polaris.models.pooling_classifier.MeanPoolingClassifier` and
:class:`~polaris.models.transformer_classifier.TransformerEncoderClassifier`.
With a third consumer arriving — the :class:`~polaris.models.embedder.TextEmbedder`
bi-encoder tower — it is factored into one shared function (ADR-0004: extract once
two or more concrete uses demand it).

Design Principles
-----------------
- A pure function on tensors: no module state, no knowledge of any model or task.
- Averages over the *real* (non-padding) positions only, using the attention
  mask, and never divides by zero for an all-padding row.
"""

from __future__ import annotations

import torch

__all__ = ["mean_pool"]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token hidden states over the real (non-padding) positions.

    Parameters
    ----------
    hidden : torch.Tensor
        Per-token hidden states of shape ``(batch_size, seq_len, dim)``.
    attention_mask : torch.Tensor
        Mask of shape ``(batch_size, seq_len)`` with ``1`` at real tokens and
        ``0`` at padding.

    Returns
    -------
    torch.Tensor
        The pooled representation of shape ``(batch_size, dim)``.

    Examples
    --------
    >>> hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])
    >>> mask = torch.tensor([[1, 1, 0]])  # third position is padding
    >>> mean_pool(hidden, mask).tolist()
    [[2.0, 2.0]]
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)
    summed = (hidden * mask).sum(dim=1)  # (B, D)
    token_counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1), avoid div-by-zero
    pooled: torch.Tensor = summed / token_counts  # (B, D)
    return pooled
