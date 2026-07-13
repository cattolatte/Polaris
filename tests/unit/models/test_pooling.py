"""Unit tests for the mask-aware mean-pool helper."""

from __future__ import annotations

import torch

from polaris.models.pooling import mean_pool

# --- correctness ---


def test_averages_over_real_tokens() -> None:
    """The pooled vector is the mean of the non-padding positions."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]]])
    mask = torch.tensor([[1, 1, 0]])  # third position is padding

    pooled = mean_pool(hidden, mask)

    assert pooled.tolist() == [[2.0, 2.0]]  # mean of 1 and 3, padding ignored


def test_padding_does_not_affect_result() -> None:
    """Extra padding columns leave the pooled value unchanged."""
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    padded_hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [99.0, 99.0]]])

    a = mean_pool(hidden, torch.tensor([[1, 1]]))
    b = mean_pool(padded_hidden, torch.tensor([[1, 1, 0]]))

    assert torch.equal(a, b)


# --- shape and edge cases ---


def test_output_shape_drops_the_sequence_axis() -> None:
    """Pooling ``(B, S, D)`` yields ``(B, D)``."""
    pooled = mean_pool(torch.randn(4, 7, 16), torch.ones(4, 7, dtype=torch.long))

    assert pooled.shape == (4, 16)


def test_all_padding_row_does_not_divide_by_zero() -> None:
    """A row with no real tokens pools to zeros rather than NaN/inf."""
    hidden = torch.randn(1, 3, 5)
    mask = torch.zeros(1, 3, dtype=torch.long)

    pooled = mean_pool(hidden, mask)

    assert torch.all(torch.isfinite(pooled))
    assert torch.all(pooled == 0.0)
