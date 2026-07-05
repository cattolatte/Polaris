"""Offline end-to-end test of the v0.11 pretrain -> fine-tune slice.

Exercises the exact thread the pretraining example uses — build a vocabulary with
a mask token, pretrain the trunk on unlabeled text, transfer it into a classifier,
fine-tune, and evaluate — on a tiny in-memory corpus. No dataset is downloaded.
"""

from __future__ import annotations

import torch

from polaris.collation import collate
from polaris.evaluation import evaluate
from polaris.models import TransformerEncoderClassifier
from polaris.pretraining import MaskedLanguageModel, pretrain
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary
from polaris.training import train
from polaris.utils import set_seed

# Unlabeled "reviews" for pretraining, and a trivially separable labeled task.
_UNLABELED: tuple[str, ...] = (
    "good great nice good",
    "bad awful terrible bad",
    "great nice good great",
    "terrible bad awful terrible",
)
_LABELED: tuple[tuple[str, int], ...] = (
    ("good great nice", 1),
    ("bad awful terrible", 0),
    ("great good", 1),
    ("awful bad", 0),
    ("nice good great", 1),
    ("terrible awful bad", 0),
)


def test_pretrain_transfer_finetune_runs_end_to_end() -> None:
    """The whole slice composes: pretrain, transfer the trunk, fine-tune, evaluate."""
    set_seed(0)

    tokenized = [text.split() for text in _UNLABELED]
    vocab = build_vocabulary(
        tokenized, pad_token="<pad>", unk_token="<unk>", mask_token="<mask>"
    )
    pad_id, mask_id = vocab.pad_id, vocab.mask_id
    assert pad_id is not None and mask_id is not None
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)

    trunk = {
        "embed_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "ff_dim": 32,
        "max_len": 16,
    }

    # Phase 1: pretrain on the unlabeled corpus.
    unlabeled_batch = collate(
        [(tokenizer.encode(text), 0) for text in _UNLABELED], pad_id=pad_id
    )
    mlm = MaskedLanguageModel(vocab_size=len(vocab), pad_id=pad_id, **trunk)  # type: ignore[arg-type]
    mlm_optimizer = torch.optim.Adam(mlm.parameters(), lr=1e-3)
    history = pretrain(
        mlm,
        mlm_optimizer,
        [unlabeled_batch],
        mask_id=mask_id,
        vocab_size=len(vocab),
        special_token_ids=(pad_id, vocab.unk_id, mask_id),
        epochs=5,
        seed=0,
    )
    assert len(history) == 5

    # Phase 2: transfer the pretrained trunk, then fine-tune on the labeled task.
    classifier = TransformerEncoderClassifier(
        vocab_size=len(vocab), num_classes=2, pad_id=pad_id, **trunk  # type: ignore[arg-type]
    )
    mlm.transfer_encoder_to(classifier)

    labeled_batch = collate(
        [(tokenizer.encode(text), label) for text, label in _LABELED], pad_id=pad_id
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-3)
    train(classifier, [labeled_batch], optimizer=optimizer, epochs=100)

    _loss, accuracy = evaluate(classifier, [labeled_batch])
    assert accuracy == 1.0
