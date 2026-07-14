"""Unit tests for the sentence-pair cross-encoder."""

from __future__ import annotations

import pytest
import torch

from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from polaris.pretraining import MaskedLanguageModel
from polaris.tokenizers.encoding import Encoding

ARCH = {
    "vocab_size": 30,
    "embed_dim": 16,
    "num_heads": 2,
    "num_layers": 1,
    "max_len": 32,
}


def _enc(*ids: int) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(f"t{i}" for i in ids))


def _batch() -> object:
    return collate_pairs(
        [(_enc(3, 4), _enc(5), 1), (_enc(6), _enc(7, 8), 0)],
        pad_id=0,
        cls_id=1,
        sep_id=2,
    )


# --- forward shape across the three tasks ---


@pytest.mark.parametrize("num_classes", [1, 2, 3])
def test_forward_shape_for_each_task(num_classes: int) -> None:
    """The head emits ``(B, num_classes)`` for rerank/gate/NLI alike."""
    model = SentencePairClassifier(num_classes=num_classes, **ARCH)  # type: ignore[arg-type]

    logits = model(_batch())  # type: ignore[arg-type]

    assert logits.shape == (2, num_classes)


# --- pooling modes ---


def test_cls_and_mean_pooling_both_work() -> None:
    """Both pooling modes produce valid logits; an unknown mode is rejected."""
    cls_model = SentencePairClassifier(num_classes=2, pooling="cls", **ARCH)  # type: ignore[arg-type]
    mean_model = SentencePairClassifier(num_classes=2, pooling="mean", **ARCH)  # type: ignore[arg-type]

    assert cls_model(_batch()).shape == (2, 2)  # type: ignore[arg-type]
    assert mean_model(_batch()).shape == (2, 2)  # type: ignore[arg-type]


def test_unknown_pooling_raises() -> None:
    """An unsupported pooling mode is a usage error."""
    with pytest.raises(ValueError, match="pooling must be"):
        SentencePairClassifier(num_classes=2, pooling="max", **ARCH)  # type: ignore[arg-type]


# --- determinism and transfer ---


def test_same_seed_reproduces_logits() -> None:
    """Construction under the same seed yields identical logits (eval mode)."""
    torch.manual_seed(0)
    a = SentencePairClassifier(num_classes=3, **ARCH).eval()  # type: ignore[arg-type]
    torch.manual_seed(0)
    b = SentencePairClassifier(num_classes=3, **ARCH).eval()  # type: ignore[arg-type]

    batch = _batch()
    assert torch.equal(a(batch), b(batch))  # type: ignore[arg-type]


def test_transfer_from_mlm_reproduces_trunk_weights() -> None:
    """A pretrained MLM trunk transfers into the pair head exactly."""
    mlm = MaskedLanguageModel(**ARCH)  # type: ignore[arg-type]
    model = SentencePairClassifier(num_classes=1, **ARCH)  # type: ignore[arg-type]

    assert not torch.equal(mlm.encoder.embedding.weight, model.encoder.embedding.weight)
    mlm.transfer_encoder_to(model)

    for pretrained, received in zip(
        mlm.encoder.state_dict().values(),
        model.encoder.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(pretrained, received)
