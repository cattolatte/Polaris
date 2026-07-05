"""Unit tests for the Polaris CLI (:mod:`polaris.cli`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from polaris.cli import app
from polaris.inference.bundle import save_bundle
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary
from polaris.version import __version__

runner = CliRunner()


def test_info_reports_version() -> None:
    """`polaris info` succeeds and prints the version."""
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    """Invoking with no command lists the available commands."""
    result = runner.invoke(app, [])

    assert "info" in result.output
    assert "predict" in result.output
    assert "serve" in result.output
    assert "console" in result.output


def _write_bundle(path: Path) -> tuple[str, str]:
    """Save a tiny bundle; return the two class names."""
    torch.manual_seed(0)
    labels = ("neg", "pos")
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
    save_bundle(
        path,
        model=MeanPoolingClassifier(**config),
        tokenizer=tokenizer,
        model_type="pooling",
        model_config=config,
        label_names=labels,
        max_length=16,
    )
    return labels


def test_predict_prints_a_label(tmp_path: Path) -> None:
    """`polaris predict` loads a bundle and prints one of its labels."""
    path = tmp_path / "model.pt"
    labels = _write_bundle(path)

    result = runner.invoke(app, ["predict", "good great", "--model", str(path)])

    assert result.exit_code == 0
    assert result.output.strip() in labels


def test_predict_probs_flag_prints_probabilities(tmp_path: Path) -> None:
    """`--probs` adds a per-class probability line for every class."""
    path = tmp_path / "model.pt"
    labels = _write_bundle(path)

    result = runner.invoke(
        app, ["predict", "good great", "--model", str(path), "--probs"]
    )

    assert result.exit_code == 0
    for name in labels:
        assert name in result.output


def test_serve_starts_the_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`polaris serve` builds the app and hands it to uvicorn (mocked)."""
    import uvicorn

    path = tmp_path / "model.pt"
    _write_bundle(path)

    served: dict[str, object] = {}

    def fake_run(app_instance: object, **kwargs: object) -> None:
        served["app"] = app_instance
        served["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(
        app, ["serve", "--model", str(path), "--host", "0.0.0.0", "--port", "9999"]
    )

    assert result.exit_code == 0
    assert served["app"] is not None
    assert served["kwargs"] == {"host": "0.0.0.0", "port": 9999}
