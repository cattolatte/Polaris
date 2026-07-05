"""Unit tests for the FastAPI serving app.

Uses FastAPI's in-process ``TestClient`` — no network, no running server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from polaris.deployment import create_app
from polaris.inference.bundle import save_bundle
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary

LABELS = ("neg", "pos")


def _client(tmp_path: Path) -> TestClient:
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
    path = tmp_path / "model.pt"
    save_bundle(
        path,
        model=MeanPoolingClassifier(**config),
        tokenizer=tokenizer,
        model_type="pooling",
        model_config=config,
        label_names=LABELS,
        max_length=16,
    )
    return TestClient(create_app(path))


# --- routes ---


def test_health_returns_ok(tmp_path: Path) -> None:
    """The health probe returns a 200 with an ok status."""
    response = _client(tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_label_and_probabilities(tmp_path: Path) -> None:
    """POST /predict classifies text and returns normalized probabilities."""
    response = _client(tmp_path).post("/predict", json={"text": "good great"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in LABELS
    assert LABELS[body["label_id"]] == body["label"]
    assert set(body["probabilities"]) == set(LABELS)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


# --- validation ---


def test_missing_text_is_rejected(tmp_path: Path) -> None:
    """A body without `text` fails request validation (422)."""
    response = _client(tmp_path).post("/predict", json={})

    assert response.status_code == 422


def test_empty_text_is_rejected(tmp_path: Path) -> None:
    """An empty `text` fails the min-length constraint (422)."""
    response = _client(tmp_path).post("/predict", json={"text": ""})

    assert response.status_code == 422
