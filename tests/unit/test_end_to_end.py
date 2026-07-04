"""Offline end-to-end integration test of the v0.4 pipeline.

Exercises the exact path the IMDB example uses — build a vocabulary, tokenize,
collate, build the model, train, and evaluate — on a tiny in-memory corpus. No
dataset is downloaded and no network is touched.
"""

from __future__ import annotations

import torch

from polaris.collation import collate
from polaris.evaluation import evaluate
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary
from polaris.training import train
from polaris.utils import set_seed

# A trivially separable sentiment task: positive and negative vocabularies.
_CORPUS: tuple[tuple[str, int], ...] = (
    ("good great nice", 1),
    ("bad awful terrible", 0),
    ("great good", 1),
    ("awful bad", 0),
    ("nice good great", 1),
    ("terrible awful bad", 0),
)


def test_pipeline_learns_a_trivial_sentiment_task() -> None:
    """The full pipeline composes and fits a tiny separable dataset."""
    set_seed(0)

    tokenized = [text.split() for text, _label in _CORPUS]
    vocab = build_vocabulary(tokenized, unk_token="<unk>", pad_token="<pad>")
    pad_id = vocab.pad_id
    assert pad_id is not None

    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    batch = collate(
        [(tokenizer.encode(text), label) for text, label in _CORPUS],
        pad_id=pad_id,
    )

    model = MeanPoolingClassifier(
        vocab_size=len(vocab), num_classes=2, embedding_dim=16, pad_id=pad_id
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    train(model, [batch], optimizer=optimizer, epochs=100)
    _loss, acc = evaluate(model, [batch])

    assert acc == 1.0
