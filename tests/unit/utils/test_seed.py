"""Unit tests for :func:`polaris.utils.seed.set_seed`.

All tests run fully offline.
"""

from __future__ import annotations

import random

import torch

from polaris.utils.seed import set_seed


def test_seed_makes_torch_draws_reproducible() -> None:
    """The same seed yields the same torch random draw."""
    set_seed(0)
    first = torch.rand(5)
    set_seed(0)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_seed_makes_python_random_reproducible() -> None:
    """The same seed yields the same Python random draw."""
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_different_seeds_differ() -> None:
    """Different seeds yield different torch draws."""
    set_seed(1)
    first = torch.rand(5)
    set_seed(2)
    second = torch.rand(5)

    assert not torch.equal(first, second)
