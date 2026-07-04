"""Unit tests for the from-scratch BPE tokenizer.

All tests run offline on tiny corpora.
"""

from __future__ import annotations

from collections.abc import Sequence

from polaris.tokenizers.bpe import BPETokenizer, train_bpe
from polaris.tokenizers.tokenizer import Tokenizer


def _corpus() -> Sequence[Sequence[str]]:
    return [["low", "lowest", "newer", "wider", "low", "newer", "new"]]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_bpe_tokenizer_is_a_tokenizer() -> None:
    """A trained BPE tokenizer satisfies the Tokenizer protocol."""
    tokenizer = train_bpe(
        _corpus(), vocab_size=50, unk_token="<unk>", pad_token="<pad>"
    )

    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(tokenizer, BPETokenizer)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_reserves_special_tokens() -> None:
    """The learned vocabulary carries the special tokens."""
    tokenizer = train_bpe(
        _corpus(), vocab_size=50, unk_token="<unk>", pad_token="<pad>"
    )

    assert tokenizer.vocabulary.unk_token == "<unk>"
    assert tokenizer.vocabulary.pad_token == "<pad>"


def test_training_is_deterministic() -> None:
    """The same corpus and size always learn the same vocabulary."""
    first = train_bpe(_corpus(), vocab_size=40, unk_token="<unk>")
    second = train_bpe(_corpus(), vocab_size=40, unk_token="<unk>")

    assert first.vocabulary == second.vocabulary


def test_frequent_pattern_merges_into_fewer_tokens() -> None:
    """A repeated pattern collapses into fewer tokens than its characters."""
    tokenizer = train_bpe([["aaaa"] * 10], vocab_size=20, unk_token="<unk>")

    tokens = tokenizer.tokenize("aaaa")

    assert len(tokens) < len("aaaa") + 1  # fewer than chars + end marker


# ---------------------------------------------------------------------------
# Subword behaviour
# ---------------------------------------------------------------------------


def test_min_frequency_still_covers_characters() -> None:
    """Words below min_frequency skip merges, but their characters stay covered."""
    tokenizer = train_bpe(
        [["aaa", "aaa", "b"]], vocab_size=20, unk_token="<unk>", min_frequency=2
    )

    encoding = tokenizer.encode("b")  # 'b' appears once, excluded from merges

    assert "<unk>" not in encoding.tokens


def test_unseen_word_splits_into_subwords_not_unknown() -> None:
    """An unseen word of known characters splits into subwords, not <unk>."""
    tokenizer = train_bpe(
        [["ab", "abc", "ab", "abc"]], vocab_size=30, unk_token="<unk>"
    )

    encoding = tokenizer.encode("abcab")  # unseen word, known characters

    assert "<unk>" not in encoding.tokens


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip() -> None:
    """Encoding then decoding reconstructs the (whitespace-normalized) text."""
    tokenizer = train_bpe(
        _corpus(), vocab_size=50, unk_token="<unk>", pad_token="<pad>"
    )

    encoding = tokenizer.encode("low newer")

    assert tokenizer.decode(encoding.ids) == "low newer"


def test_encode_aligns_ids_and_tokens() -> None:
    """Encoding produces aligned ids and subword tokens."""
    tokenizer = train_bpe(_corpus(), vocab_size=50, unk_token="<unk>")

    encoding = tokenizer.encode("newer")

    assert len(encoding.ids) == len(encoding.tokens)
