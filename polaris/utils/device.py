"""Device selection for training and inference.

Chooses the best available PyTorch device so training can use an Apple Silicon
GPU (MPS) or a CUDA GPU when present, and falls back to CPU otherwise.
"""

from __future__ import annotations

import torch

__all__ = ["resolve_device", "module_device"]


def resolve_device(prefer: str | None = None) -> torch.device:
    """Return a torch device, preferring MPS, then CUDA, then CPU.

    Parameters
    ----------
    prefer : str, optional
        An explicit device string (e.g. ``"cpu"``, ``"mps"``, ``"cuda"``). When
        given it is used as-is; otherwise the best available device is chosen.

    Returns
    -------
    torch.device
        The selected device.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def module_device(model: torch.nn.Module) -> torch.device:
    """Return the device a model's parameters live on.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.

    Returns
    -------
    torch.device
        The device of the first parameter, or CPU if the model has none.
    """
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")
