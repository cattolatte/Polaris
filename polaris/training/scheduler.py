"""A learning-rate schedule with warmup, implemented from scratch.

Warmup — ramping the learning rate up from near zero over the first steps before
decaying it — is what lets a transformer train stably from scratch. We compute it
by hand rather than hide it behind a library call (see ADR-0003).
"""

from __future__ import annotations

import math

import torch

__all__ = ["WarmupSchedule"]


class WarmupSchedule:
    """Linear warmup, then linear or cosine decay to ``min_lr``.

    Call :meth:`step` once per optimizer step; it sets the learning rate on every
    parameter group and advances the internal step counter.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate is scheduled.
    base_lr : float
        The peak learning rate reached at the end of warmup.
    warmup_steps : int
        Number of steps to ramp linearly from ~0 to ``base_lr``.
    total_steps : int
        Total number of scheduled steps.
    decay : str, default "cosine"
        The post-warmup decay shape: ``"cosine"`` or ``"linear"``.
    min_lr : float, default 0.0
        The learning rate reached at ``total_steps``.

    Raises
    ------
    ValueError
        If the step counts are invalid or ``decay`` is unknown.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        decay: str = "cosine",
        min_lr: float = 0.0,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) cannot exceed "
                f"total_steps ({total_steps})"
            )
        if decay not in ("cosine", "linear"):
            raise ValueError(f"decay must be 'cosine' or 'linear', got {decay!r}")

        self._optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay = decay
        self.min_lr = min_lr
        self._step_count = 0

    def lr_at(self, step: int) -> float:
        """Return the learning rate for a given step (pure computation)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        if self.decay == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0 - progress
        return self.min_lr + (self.base_lr - self.min_lr) * factor

    def step(self) -> float:
        """Set the current learning rate on the optimizer and advance.

        Returns
        -------
        float
            The learning rate that was applied.
        """
        lr = self.lr_at(self._step_count)
        for group in self._optimizer.param_groups:
            group["lr"] = lr
        self._step_count += 1
        return lr
