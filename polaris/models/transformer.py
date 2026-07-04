"""Transformer encoder building blocks, implemented from scratch.

Layer normalization, sinusoidal positional encoding, and a transformer encoder
block — all written on tensor primitives (see ADR-0003). We use these rather
than ``nn.LayerNorm`` / ``nn.TransformerEncoderLayer`` because implementing them
is the point.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from polaris.models.attention import MultiHeadSelfAttention

__all__ = [
    "LayerNorm",
    "SinusoidalPositionalEncoding",
    "TransformerEncoderBlock",
]


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, from scratch.

    Normalizes each vector to zero mean and unit variance, then applies a
    learned per-feature scale and shift.

    Parameters
    ----------
    dim : int
        The size of the last dimension to normalize.
    eps : float, default 1e-5
        Added to the variance for numerical stability.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * normalized + self.bias


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings, added to token embeddings.

    Parameters
    ----------
    embed_dim : int
        The model dimension.
    max_len : int, default 5000
        The maximum sequence length supported.
    """

    pe: torch.Tensor

    def __init__(self, *, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        # A buffer moves with .to(device) but is not a learnable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x`` of shape ``(batch, seq, embed)``."""
        return x + self.pe[: x.shape[1]]


class TransformerEncoderBlock(nn.Module):
    """A single pre-norm transformer encoder block.

    Self-attention and a position-wise feed-forward network, each wrapped in a
    residual connection with layer normalization applied *before* the sublayer
    (the pre-norm formulation, which trains stably without learning-rate warmup;
    the original transformer used post-norm).

    Parameters
    ----------
    embed_dim : int
        The model dimension.
    num_heads : int
        The number of attention heads.
    ff_dim : int
        The hidden size of the feed-forward network.
    dropout : float, default 0.0
        Dropout probability applied to each sublayer's output.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the block over ``x`` of shape ``(batch, seq, embed)``."""
        attended = self.attention(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.dropout(attended)
        fed: torch.Tensor = self.feed_forward(self.norm2(x))
        x = x + self.dropout(fed)
        return x
