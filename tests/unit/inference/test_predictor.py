"""Unit tests for the Predictor."""

from __future__ import annotations

import pytest

from polaris.errors import PolarisError
from polaris.inference.predictor import Prediction, Predictor
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary

LABELS = ("neg", "pos")


def _predictor() -> Predictor:
    vocab = build_vocabulary(
        [["good", "great"], ["bad", "awful"]], unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    model = MeanPoolingClassifier(
        vocab_size=len(vocab), num_classes=2, embedding_dim=8, pad_id=vocab.pad_id
    )
    return Predictor(model, tokenizer, label_names=LABELS, max_length=16)


# --- prediction contract ---


def test_predict_returns_a_valid_prediction() -> None:
    """A single prediction names a known label with a matching label id."""
    prediction = _predictor().predict("good great")

    assert prediction.label in LABELS
    assert LABELS[prediction.label_id] == prediction.label


def test_probabilities_cover_all_classes_and_sum_to_one() -> None:
    """Per-class probabilities are keyed by class name and normalized."""
    prediction = _predictor().predict("good bad")

    assert set(prediction.probabilities) == set(LABELS)
    assert prediction.probabilities[prediction.label] == max(
        prediction.probabilities.values()
    )
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_predict_batch_returns_one_prediction_per_text() -> None:
    """Batch prediction returns aligned results, in order."""
    predictions = _predictor().predict_batch(["good great", "bad awful", "good"])

    assert len(predictions) == 3
    assert all(isinstance(p, Prediction) for p in predictions)


def test_empty_batch_returns_empty_list() -> None:
    """Predicting no texts yields no predictions (no model call)."""
    assert _predictor().predict_batch([]) == []


# --- invariants ---


def test_prediction_probabilities_are_read_only() -> None:
    """The probabilities mapping cannot be mutated after construction."""
    prediction = _predictor().predict("good")

    with pytest.raises(TypeError):
        prediction.probabilities["neg"] = 0.5  # type: ignore[index]


def test_predictor_requires_a_padding_token() -> None:
    """A vocabulary without a pad id cannot back a predictor."""
    vocab = build_vocabulary([["good", "bad"]], unk_token="<unk>")
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    model = MeanPoolingClassifier(vocab_size=len(vocab), num_classes=2, embedding_dim=8)

    with pytest.raises(PolarisError, match="padding token"):
        Predictor(model, tokenizer, label_names=LABELS)
