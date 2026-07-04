"""Capture the runtime environment, for reproducibility.

Recording the library versions and platform alongside a run means a result can be
understood and reproduced later, not just trusted.
"""

from __future__ import annotations

import platform
import sys

from polaris.version import __version__

__all__ = ["capture_environment"]


def capture_environment() -> dict[str, str]:
    """Return a record of the runtime environment.

    Returns
    -------
    dict[str, str]
        The Polaris, Python, PyTorch, and platform versions. ``torch`` is
        reported as ``"not installed"`` if the optional dependency is absent.
    """
    environment = {
        "polaris": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        environment["torch"] = torch.__version__
    except ImportError:
        environment["torch"] = "not installed"
    return environment
