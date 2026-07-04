"""Attention, implemented from scratch.

The heart of the transformer, written directly on tensor primitives so it can be
read and understood line by line (see ADR-0003). Nothing here uses
``nn.MultiheadAttention`` or ``nn.functional.scaled_dot_product_attention`` — the
point is to build it ourselves.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["scaled_dot_product_attention", "MultiHeadSelfAttention"]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``softmax(Q Kᵀ / √d_k) V``.

    Parameters
    ----------
    query : torch.Tensor
        Shape ``(..., seq_q, d_k)``.
    key : torch.Tensor
        Shape ``(..., seq_k, d_k)``.
    value : torch.Tensor
        Shape ``(..., seq_k, d_v)``.
    mask : torch.Tensor, optional
        Broadcastable to the scores shape ``(..., seq_q, seq_k)``. Positions
        where ``mask == 0`` are not attended to (set to ``-inf`` before softmax).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The attention output ``(..., seq_q, d_v)`` and the attention weights
        ``(..., seq_q, seq_k)``.
    """
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over a sequence.

    Projects the input to queries, keys, and values; splits them into
    ``num_heads`` independent heads; applies scaled dot-product attention per
    head; then concatenates the heads and projects back.

    Parameters
    ----------
    embed_dim : int
        The model dimension. Must be divisible by ``num_heads``.
    num_heads : int
        The number of attention heads.

    Raises
    ------
    ValueError
        If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(self, *, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = (
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
            raise ValueError(msg)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, seq, embed) -> (batch, heads, seq, head_dim)."""
        batch, seq = tensor.shape[0], tensor.shape[1]
        reshaped = tensor.view(batch, seq, self.num_heads, self.head_dim)
        return reshaped.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, head_dim) -> (batch, seq, embed)."""
        batch, seq = tensor.shape[0], tensor.shape[2]
        merged = tensor.transpose(1, 2).contiguous()
        return merged.view(batch, seq, self.embed_dim)

    def forward(
        self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq, embed_dim)``.
        key_padding_mask : torch.Tensor, optional
            Shape ``(batch, seq)``; ``1`` for real tokens, ``0`` for padding.
            Padding positions are not attended to.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, seq, embed_dim)``.
        """
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        mask = None
        if key_padding_mask is not None:
            # (batch, seq) -> (batch, 1, 1, seq): broadcast over heads and queries.
            mask = key_padding_mask[:, None, None, :]

        attended, _weights = scaled_dot_product_attention(query, key, value, mask=mask)
        output: torch.Tensor = self.out_proj(self._merge_heads(attended))
        return output
