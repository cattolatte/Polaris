"""Unit tests for the interactive console REPL (driven via ``onecmd``, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polaris.console import PolarisConsole
from polaris.inference.bundle import save_bundle
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary

LABELS = ("neg", "pos")


def _write_bundle(path: Path) -> None:
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
    save_bundle(
        path,
        model=MeanPoolingClassifier(**config),
        tokenizer=tokenizer,
        model_type="pooling",
        model_config=config,
        label_names=LABELS,
        max_length=16,
    )


# --- load ---


def test_load_then_predict_prints_a_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Loading a bundle enables prediction; a known label is printed."""
    path = tmp_path / "model.pt"
    _write_bundle(path)
    console = PolarisConsole()

    console.onecmd(f"load {path}")
    console.onecmd("predict good great")

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == f"loaded: {path}"
    assert lines[-1] in LABELS


def test_load_missing_file_is_friendly(capsys: pytest.CaptureFixture[str]) -> None:
    """A bad path prints a hint, not a traceback."""
    PolarisConsole().onecmd("load /nowhere/model.pt")

    assert "no such file" in capsys.readouterr().out


# --- predict guards ---


def test_predict_without_model_hints_at_load(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Predicting before loading points the user at `load`."""
    PolarisConsole().onecmd("predict some text")

    assert "no model loaded" in capsys.readouterr().out


def test_probs_toggle_adds_probabilities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After `probs`, predictions include one probability line per class."""
    path = tmp_path / "model.pt"
    _write_bundle(path)
    console = PolarisConsole()
    console.onecmd(f"load {path}")
    console.onecmd("probs")
    capsys.readouterr()

    console.onecmd("predict good great")

    out = capsys.readouterr().out
    for name in LABELS:
        assert f"{name}:" in out


# --- shell behavior ---


def test_exit_and_quit_end_the_loop() -> None:
    """`exit`, `quit`, and Ctrl-D all return True (stop signal)."""
    console = PolarisConsole()

    assert console.onecmd("exit") is True
    assert console.onecmd("quit") is True
    assert console.onecmd("EOF") is True


def test_unknown_command_points_at_help(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown command suggests `help` instead of erroring."""
    PolarisConsole().onecmd("frobnicate everything")

    assert "help" in capsys.readouterr().out


def test_empty_line_does_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty line is a no-op (cmd's default would repeat the last command)."""
    console = PolarisConsole()

    assert console.emptyline() is False
    assert capsys.readouterr().out == ""
