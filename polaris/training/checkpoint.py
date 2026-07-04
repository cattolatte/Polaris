"""Save and restore training checkpoints.

A checkpoint bundles the model weights, optionally the optimizer state, and a
small metadata dictionary (e.g. the epoch and best metric), so a run can keep the
*best* model rather than the last one and be resumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint to ``path``.

    Parameters
    ----------
    path : str or Path
        Destination file. Parent directories are created if needed.
    model : nn.Module
        The model whose ``state_dict`` is saved.
    optimizer : torch.optim.Optimizer, optional
        If given, its ``state_dict`` is saved too (for resuming).
    metadata : dict, optional
        Small JSON-like extras (e.g. epoch, best metric).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint into ``model`` (and optionally ``optimizer``).

    Parameters
    ----------
    path : str or Path
        The checkpoint file to load.
    model : nn.Module
        The model to load weights into (modified in place).
    optimizer : torch.optim.Optimizer, optional
        If given and the checkpoint has optimizer state, it is restored.

    Returns
    -------
    dict
        The saved metadata dictionary.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    metadata: dict[str, Any] = payload.get("metadata", {})
    return metadata
