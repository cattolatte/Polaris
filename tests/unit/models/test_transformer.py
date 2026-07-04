"""Unit tests for :mod:`polaris.models.transformer`.

All tests run fully offline on tiny tensors.
"""

from __future__ import annotations

import torch

from polaris.models.transformer import (
    LayerNorm,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)

# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


def test_layer_norm_normalizes_last_dimension() -> None:
    """With default affine params, output has ~zero mean and ~unit std."""
    norm = LayerNorm(8)
    x = torch.randn(4, 8) * 5.0 + 3.0

    out = norm(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(4), atol=1e-3)


def test_layer_norm_preserves_shape() -> None:
    """Layer norm does not change the tensor shape."""
    norm = LayerNorm(6)
    x = torch.randn(2, 3, 6)

    assert norm(x).shape == (2, 3, 6)


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------


def test_positional_encoding_preserves_shape_and_adds_signal() -> None:
    """Positional encoding keeps the shape and actually changes the input."""
    encoding = SinusoidalPositionalEncoding(embed_dim=8, max_len=50)
    x = torch.zeros(2, 5, 8)

    out = encoding(x)

    assert out.shape == (2, 5, 8)
    assert not torch.allclose(out, x)


def test_positional_encoding_is_deterministic() -> None:
    """The encoding is fixed (not learned) and identical across instances."""
    first = SinusoidalPositionalEncoding(embed_dim=8, max_len=10)
    second = SinusoidalPositionalEncoding(embed_dim=8, max_len=10)

    assert torch.equal(first.pe, second.pe)


# ---------------------------------------------------------------------------
# TransformerEncoderBlock
# ---------------------------------------------------------------------------


def test_block_preserves_shape() -> None:
    """A block maps (batch, seq, embed) to the same shape."""
    block = TransformerEncoderBlock(embed_dim=8, num_heads=2, ff_dim=16)
    x = torch.randn(3, 5, 8)

    assert block(x).shape == (3, 5, 8)


def test_block_runs_with_padding_mask() -> None:
    """A block accepts a padding mask and produces finite outputs."""
    block = TransformerEncoderBlock(embed_dim=8, num_heads=2, ff_dim=16)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    out = block(x, key_padding_mask=mask)

    assert out.shape == (2, 4, 8)
    assert bool(torch.isfinite(out).all())
