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


def test_train_bpe_reserves_pair_special_tokens_first() -> None:
    """cls/sep/mask are reserved contiguously before any learned symbols."""
    tokenizer = train_bpe(
        [["good", "movie"], ["good", "film"]],
        vocab_size=50,
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        cls_token="<cls>",
        sep_token="<sep>",
    )
    vocab = tokenizer.vocabulary
    assert (vocab.pad_id, vocab.unk_id, vocab.mask_id, vocab.cls_id, vocab.sep_id) == (
        0,
        1,
        2,
        3,
        4,
    )


def test_bpe_dict_round_trip_preserves_encoding() -> None:
    """to_dict / from_dict reconstructs a tokenizer that encodes identically."""
    tokenizer = train_bpe(
        _corpus(), vocab_size=50, unk_token="<unk>", pad_token="<pad>"
    )

    restored = BPETokenizer.from_dict(tokenizer.to_dict())

    text = "the cat sat"
    assert restored.encode(text).ids == tokenizer.encode(text).ids
    assert restored.decode(tokenizer.encode(text).ids) == tokenizer.decode(
        tokenizer.encode(text).ids
    )


def test_bpe_save_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A saved tokenizer loads back and encodes/decodes identically."""
    tokenizer = train_bpe(
        _corpus(), vocab_size=50, unk_token="<unk>", pad_token="<pad>"
    )
    path = tmp_path / "tok.json"
    tokenizer.save(path)

    restored = BPETokenizer.load(path)

    text = "the dog sat"
    assert restored.encode(text).ids == tokenizer.encode(text).ids
    assert restored.vocabulary == tokenizer.vocabulary
