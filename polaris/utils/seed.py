"""Deterministic seeding for reproducible runs.

The first concrete home for Polaris' reproducibility goal: seed the random
number generators a training run depends on so that the same inputs produce the
same results.
"""

from __future__ import annotations

import random

import torch

__all__ = ["set_seed"]


def set_seed(seed: int) -> None:
    """Seed the Python and PyTorch random number generators.

    Parameters
    ----------
    seed : int
        The seed value applied to both generators.

    Examples
    --------
    >>> import torch
    >>> set_seed(0)
    >>> a = torch.rand(3)
    >>> set_seed(0)
    >>> b = torch.rand(3)
    >>> bool(torch.equal(a, b))
    True
    """
    random.seed(seed)
    torch.manual_seed(seed)
