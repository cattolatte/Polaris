"""Unit tests for the InfoNCE contrastive loss."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from polaris.training.losses import info_nce_loss


def _normalized(*rows: list[float]) -> torch.Tensor:
    return F.normalize(torch.tensor(rows), dim=-1)


# --- in-batch-negative math ---


def test_orthonormal_pairs_match_the_analytic_cross_entropy() -> None:
    """On an orthonormal hand example the loss equals the softmax cross-entropy."""
    embeddings = _normalized([1.0, 0.0], [0.0, 1.0])  # identity, already unit norm

    loss = info_nce_loss(embeddings, embeddings, temperature=1.0, symmetric=True)

    # Row i logits are [1,0] (or [0,1]); target i => -log(softmax([1,0])[0]).
    expected = -math.log(math.exp(1.0) / (math.exp(1.0) + math.exp(0.0)))
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_aligned_positive_gives_near_zero_loss() -> None:
    """When each positive equals its anchor, the loss is very small."""
    anchor = _normalized([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 0.0, 1.0])

    loss = info_nce_loss(anchor, anchor.clone(), temperature=0.05)

    assert loss.item() < 0.05


# --- hard negatives ---


def test_hard_negatives_raise_the_loss() -> None:
    """Appending mined hard negatives to the denominator increases the loss."""
    torch.manual_seed(0)
    anchor = F.normalize(torch.randn(4, 8), dim=-1)
    positive = F.normalize(torch.randn(4, 8), dim=-1)
    hard = F.normalize(torch.randn(4, 5, 8), dim=-1)

    without = info_nce_loss(anchor, positive, extra_negatives=None)
    with_hard = info_nce_loss(anchor, positive, extra_negatives=hard)

    assert with_hard.item() > without.item()


# --- symmetry ---


def test_symmetric_differs_from_one_directional() -> None:
    """Averaging both directions gives a different value than one direction."""
    torch.manual_seed(0)
    anchor = F.normalize(torch.randn(4, 8), dim=-1)
    positive = F.normalize(torch.randn(4, 8), dim=-1)

    both = info_nce_loss(anchor, positive, symmetric=True)
    one = info_nce_loss(anchor, positive, symmetric=False)

    assert not torch.isclose(both, one)


# --- gradients and validation ---


def test_loss_is_differentiable() -> None:
    """Gradients flow back to the embeddings."""
    anchor = F.normalize(torch.randn(3, 6), dim=-1).requires_grad_(True)
    positive = F.normalize(torch.randn(3, 6), dim=-1).requires_grad_(True)

    info_nce_loss(anchor, positive).backward()

    assert anchor.grad is not None
    assert torch.any(anchor.grad != 0.0)


def test_shape_mismatch_raises() -> None:
    """Anchor and positive must share a shape."""
    with pytest.raises(ValueError, match="share a shape"):
        info_nce_loss(torch.randn(3, 8), torch.randn(3, 4))


def test_non_positive_temperature_raises() -> None:
    """Temperature must be positive."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        info_nce_loss(torch.randn(2, 4), torch.randn(2, 4), temperature=0.0)
