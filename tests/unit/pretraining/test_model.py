"""Unit tests for the masked-language model."""

from __future__ import annotations

import torch

from polaris.models import TransformerEncoderClassifier
from polaris.pretraining.masking import mask_tokens
from polaris.pretraining.model import MaskedLanguageModel


def _masked_batch(vocab_size: int) -> object:
    ids = torch.randint(3, vocab_size, (2, 6))
    attention = torch.ones(2, 6, dtype=torch.long)
    return mask_tokens(
        ids,
        attention,
        mask_id=2,
        vocab_size=vocab_size,
        generator=torch.Generator().manual_seed(0),
    )


# --- forward ---


def test_forward_returns_per_token_vocab_logits() -> None:
    """The model emits ``(batch, seq, vocab)`` logits, one distribution per token."""
    vocab_size = 20
    model = MaskedLanguageModel(
        vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=1, max_len=8
    )

    logits = model(_masked_batch(vocab_size))  # type: ignore[arg-type]

    assert logits.shape == (2, 6, vocab_size)


# --- trunk transfer ---

# A matching trunk architecture, shared by the model and the classifier below.
VOCAB_SIZE = 20


def _model_and_classifier() -> tuple[MaskedLanguageModel, TransformerEncoderClassifier]:
    model = MaskedLanguageModel(
        vocab_size=VOCAB_SIZE, embed_dim=16, num_heads=2, num_layers=1, max_len=8
    )
    classifier = TransformerEncoderClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=2,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_len=8,
    )
    return model, classifier


def test_transfer_copies_trunk_into_classifier() -> None:
    """Transferring overwrites the classifier's trunk with the pretrained one."""
    model, classifier = _model_and_classifier()

    # Independently initialized trunks differ (random embedding/attention weights).
    assert not torch.equal(
        model.encoder.embedding.weight, classifier.encoder.embedding.weight
    )

    model.transfer_encoder_to(classifier)

    # After transfer every trunk parameter matches the pretrained model's.
    for pretrained, received in zip(
        model.encoder.state_dict().values(),
        classifier.encoder.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(pretrained, received)


def test_transfer_leaves_classifier_head_untouched() -> None:
    """Only the shared trunk transfers; the classification head is preserved."""
    model, classifier = _model_and_classifier()

    head_before = classifier.classifier.weight.clone()
    model.transfer_encoder_to(classifier)

    assert torch.equal(head_before, classifier.classifier.weight)
