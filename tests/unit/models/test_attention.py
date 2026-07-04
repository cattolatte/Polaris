"""Unit tests for :mod:`polaris.models.attention`.

All tests run fully offline on tiny tensors. They assert shapes and invariants
(weights form a distribution, masking is respected), never exact float values.
"""

from __future__ import annotations

import pytest
import torch

from polaris.models.attention import (
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
)

# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------


def test_output_and_weight_shapes() -> None:
    """Output and weights have the expected shapes."""
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 6)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (2, 3, 6)
    assert weights.shape == (2, 3, 5)


def test_weights_sum_to_one() -> None:
    """Attention weights form a probability distribution over keys."""
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 4)

    _output, weights = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_mask_zeroes_attention_on_masked_keys() -> None:
    """Masked key positions receive zero attention weight."""
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 4)
    mask = torch.tensor([[[1, 1, 0]]])  # (1, 1, 3): mask the last key

    _output, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    assert torch.allclose(weights[..., 2], torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2), atol=1e-5)


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention
# ---------------------------------------------------------------------------


def test_multihead_preserves_shape() -> None:
    """Multi-head attention returns the input shape."""
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(3, 5, 8)

    assert attention(x).shape == (3, 5, 8)


def test_multihead_requires_divisible_dims() -> None:
    """embed_dim must be divisible by num_heads."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(embed_dim=8, num_heads=3)


def test_single_head_runs() -> None:
    """A single head is a valid configuration."""
    attention = MultiHeadSelfAttention(embed_dim=6, num_heads=1)
    x = torch.randn(2, 3, 6)

    assert attention(x).shape == (2, 3, 6)


def test_multihead_runs_with_padding_mask() -> None:
    """Padding-masked attention produces finite outputs of the right shape."""
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    output = attention(x, key_padding_mask=mask)

    assert output.shape == (2, 4, 8)
    assert bool(torch.isfinite(output).all())
