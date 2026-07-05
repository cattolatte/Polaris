"""Unit tests for saving and loading model bundles."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polaris.errors import PolarisError
from polaris.inference.bundle import load_bundle, save_bundle
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary

LABELS = ("neg", "pos")


def _model_and_tokenizer() -> tuple[MeanPoolingClassifier, WhitespaceTokenizer, dict]:
    torch.manual_seed(0)
    vocab = build_vocabulary(
        [["good", "great"], ["bad", "awful"]], unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    config = {
        "vocab_size": len(vocab),
        "num_classes": 2,
        "embedding_dim": 8,
        "pad_id": vocab.pad_id,
    }
    model = MeanPoolingClassifier(**config)
    return model, tokenizer, config


def _save(path: Path) -> None:
    model, tokenizer, config = _model_and_tokenizer()
    save_bundle(
        path,
        model=model,
        tokenizer=tokenizer,
        model_type="pooling",
        model_config=config,
        label_names=LABELS,
        max_length=16,
    )


# --- round trip ---


def test_bundle_round_trip_reproduces_predictions(tmp_path: Path) -> None:
    """A reloaded bundle predicts identically to the original model."""
    model, tokenizer, config = _model_and_tokenizer()
    from polaris.inference.predictor import Predictor

    original = Predictor(model, tokenizer, label_names=LABELS, max_length=16)

    path = tmp_path / "model.pt"
    save_bundle(
        path,
        model=model,
        tokenizer=tokenizer,
        model_type="pooling",
        model_config=config,
        label_names=LABELS,
        max_length=16,
    )
    restored = load_bundle(path)

    for text in ["good great", "bad awful", "good bad awful"]:
        before = original.predict(text)
        after = restored.predict(text)
        assert after.label == before.label
        assert after.probabilities[after.label] == pytest.approx(
            before.probabilities[before.label]
        )


def test_load_returns_usable_predictor(tmp_path: Path) -> None:
    """Loading a bundle yields a predictor that classifies raw text."""
    path = tmp_path / "model.pt"
    _save(path)

    prediction = load_bundle(path).predict("good great")

    assert prediction.label in LABELS


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """Saving to a nested path creates the intermediate directories."""
    path = tmp_path / "nested" / "dir" / "model.pt"
    _save(path)

    assert path.exists()


# --- guards ---


def test_save_rejects_non_whitespace_tokenizer(tmp_path: Path) -> None:
    """Only whitespace tokenizers can be bundled (for now)."""
    model, _tokenizer, config = _model_and_tokenizer()

    class FakeTokenizer:
        pass

    with pytest.raises(PolarisError, match="WhitespaceTokenizer"):
        save_bundle(
            tmp_path / "model.pt",
            model=model,
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            model_type="pooling",
            model_config=config,
            label_names=LABELS,
        )


def test_load_rejects_unsupported_tokenizer_type(tmp_path: Path) -> None:
    """A bundle naming an unknown tokenizer type is rejected on load."""
    path = tmp_path / "model.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["tokenizer_type"] = "bpe"
    torch.save(payload, path)

    with pytest.raises(PolarisError, match="tokenizer_type"):
        load_bundle(path)
