"""Unit tests for :mod:`polaris.utils.device`.

All tests run fully offline. The auto-selection branches are exercised by
patching the availability checks, so no GPU is required.
"""

from __future__ import annotations

from unittest.mock import patch

from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.utils.device import module_device, resolve_device

_MPS = "torch.backends.mps.is_available"
_CUDA = "torch.cuda.is_available"


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_explicit_choice_is_honored() -> None:
    """An explicit device string is used as-is."""
    assert resolve_device("cpu").type == "cpu"


def test_prefers_mps_when_available() -> None:
    """MPS (Apple Silicon) is chosen first when available."""
    with patch(_MPS, return_value=True):
        assert resolve_device().type == "mps"


def test_falls_back_to_cuda_when_no_mps() -> None:
    """CUDA is chosen when MPS is unavailable."""
    with (
        patch(_MPS, return_value=False),
        patch(_CUDA, return_value=True),
    ):
        assert resolve_device().type == "cuda"


def test_falls_back_to_cpu_when_no_accelerator() -> None:
    """CPU is chosen when neither MPS nor CUDA is available."""
    with (
        patch(_MPS, return_value=False),
        patch(_CUDA, return_value=False),
    ):
        assert resolve_device().type == "cpu"


# ---------------------------------------------------------------------------
# module_device
# ---------------------------------------------------------------------------


def test_module_device_reads_parameter_device() -> None:
    """The model's parameter device is reported (CPU by default)."""
    model = MeanPoolingClassifier(vocab_size=5, num_classes=2)

    assert module_device(model).type == "cpu"
