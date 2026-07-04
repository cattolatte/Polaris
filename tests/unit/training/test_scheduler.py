"""Unit tests for :class:`polaris.training.scheduler.WarmupSchedule`."""

from __future__ import annotations

import pytest
import torch

from polaris.training.scheduler import WarmupSchedule


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([parameter], lr=0.0)


# ---------------------------------------------------------------------------
# Schedule shape
# ---------------------------------------------------------------------------


def test_warmup_ramps_up_then_decays() -> None:
    """The LR rises during warmup and falls afterward toward min_lr."""
    schedule = WarmupSchedule(
        _optimizer(),
        base_lr=1.0,
        warmup_steps=10,
        total_steps=100,
        decay="linear",
    )

    start = schedule.lr_at(0)
    peak = schedule.lr_at(9)
    late = schedule.lr_at(55)
    end = schedule.lr_at(99)

    assert start < peak
    assert peak == pytest.approx(1.0)
    assert end < late < peak
    assert end < 0.05


def test_no_warmup_starts_at_base_lr() -> None:
    """With zero warmup, step 0 is already at the peak learning rate."""
    schedule = WarmupSchedule(_optimizer(), base_lr=0.5, warmup_steps=0, total_steps=10)

    assert schedule.lr_at(0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Optimizer integration
# ---------------------------------------------------------------------------


def test_step_sets_optimizer_learning_rate() -> None:
    """``step`` writes the current LR onto the optimizer and advances."""
    optimizer = _optimizer()
    schedule = WarmupSchedule(optimizer, base_lr=1.0, warmup_steps=2, total_steps=10)

    first = schedule.step()
    second = schedule.step()

    assert optimizer.param_groups[0]["lr"] == second
    assert first < second  # still warming up


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_warmup_greater_than_total_raises() -> None:
    """Warmup cannot exceed the total number of steps."""
    with pytest.raises(ValueError, match="warmup_steps"):
        WarmupSchedule(_optimizer(), base_lr=1.0, warmup_steps=20, total_steps=10)


def test_unknown_decay_raises() -> None:
    """An unknown decay shape is rejected."""
    with pytest.raises(ValueError, match="decay"):
        WarmupSchedule(
            _optimizer(),
            base_lr=1.0,
            warmup_steps=1,
            total_steps=10,
            decay="quadratic",
        )
