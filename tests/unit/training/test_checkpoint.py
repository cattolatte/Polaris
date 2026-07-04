"""Unit tests for :mod:`polaris.training.checkpoint`.

Checkpoints are written to a pytest temp directory — offline, no network.
"""

from __future__ import annotations

from pathlib import Path

import torch

from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.training.checkpoint import load_checkpoint, save_checkpoint


def _model() -> MeanPoolingClassifier:
    return MeanPoolingClassifier(vocab_size=5, num_classes=2, embedding_dim=8)


def test_roundtrip_restores_weights_and_metadata(tmp_path: Path) -> None:
    """Saving then loading restores the model weights and metadata exactly."""
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)  # perturb so weights are non-default

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, metadata={"epoch": 3})

    restored = _model()
    metadata = load_checkpoint(path, model=restored)

    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)
    assert metadata["epoch"] == 3


def test_optimizer_state_roundtrips(tmp_path: Path) -> None:
    """Optimizer state is saved and restored when provided."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer)

    restored = _model()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    metadata = load_checkpoint(path, model=restored, optimizer=restored_optimizer)

    assert metadata == {}


def test_creates_parent_directories(tmp_path: Path) -> None:
    """Saving into a missing subdirectory creates it."""
    path = tmp_path / "nested" / "dir" / "checkpoint.pt"

    save_checkpoint(path, model=_model())

    assert path.exists()
