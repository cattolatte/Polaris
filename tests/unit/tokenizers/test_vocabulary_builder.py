"""Unit tests for :func:`polaris.tokenizers.vocabulary_builder.build_vocabulary`.

Design Principles
------------------
- Tests exercise only the public behaviour of ``build_vocabulary``:
  frequency ordering, determinism, special-token reservation, filtering,
  size limiting, and edge cases.
- All tests run fully offline with no network access.
"""

from __future__ import annotations

import pytest

from polaris.tokenizers.vocabulary import Vocabulary
from polaris.tokenizers.vocabulary_builder import build_vocabulary

# ---------------------------------------------------------------------------
# Basic building
# ---------------------------------------------------------------------------


def test_builds_vocabulary_from_token_sequences() -> None:
    """A vocabulary is built from the distinct tokens in the corpus."""
    vocabulary = build_vocabulary([["a", "b"], ["a", "c"]])

    assert isinstance(vocabulary, Vocabulary)
    assert set(vocabulary.token_to_id) == {"a", "b", "c"}


def test_ids_are_contiguous_from_zero() -> None:
    """Assigned ids are contiguous starting at zero."""
    vocabulary = build_vocabulary([["a", "b", "c"]])

    assert sorted(vocabulary.token_to_id.values()) == [0, 1, 2]


def test_more_frequent_tokens_get_smaller_ids() -> None:
    """Tokens are ordered by descending frequency."""
    vocabulary = build_vocabulary([["a", "a", "a"], ["b", "b"], ["c"]])

    assert vocabulary.lookup_id("a") < vocabulary.lookup_id("b")
    assert vocabulary.lookup_id("b") < vocabulary.lookup_id("c")


def test_ties_are_broken_alphabetically_for_determinism() -> None:
    """Equal-frequency tokens are ordered by token text, regardless of input order."""
    first = build_vocabulary([["b", "a"]])
    second = build_vocabulary([["a", "b"]])

    assert first.token_to_id == second.token_to_id
    assert first.lookup_id("a") < first.lookup_id("b")


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def test_special_tokens_are_reserved_first() -> None:
    """Padding and unknown tokens take the lowest ids, padding first."""
    vocabulary = build_vocabulary(
        [["hello", "hello"]], pad_token="<pad>", unk_token="<unk>"
    )

    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1
    assert vocabulary.lookup_id("hello") == 2


def test_only_unk_reserved_when_pad_absent() -> None:
    """When only unk is given, it takes id zero."""
    vocabulary = build_vocabulary([["a"]], unk_token="<unk>")

    assert vocabulary.unk_id == 0
    assert vocabulary.pad_id is None


def test_special_tokens_are_not_counted_as_corpus_tokens() -> None:
    """A corpus occurrence of a special token does not create a duplicate entry."""
    vocabulary = build_vocabulary([["<unk>", "a"]], unk_token="<unk>")

    assert vocabulary.unk_id == 0
    assert vocabulary.lookup_id("a") == 1
    assert len(vocabulary) == 2


# ---------------------------------------------------------------------------
# min_frequency
# ---------------------------------------------------------------------------


def test_min_frequency_filters_rare_tokens() -> None:
    """Tokens below the frequency threshold are dropped."""
    vocabulary = build_vocabulary([["a", "a", "b"]], min_frequency=2)

    assert "a" in vocabulary
    assert "b" not in vocabulary


def test_min_frequency_below_one_raises() -> None:
    """A minimum frequency below one is rejected."""
    with pytest.raises(ValueError, match="min_frequency"):
        build_vocabulary([["a"]], min_frequency=0)


# ---------------------------------------------------------------------------
# max_size
# ---------------------------------------------------------------------------


def test_max_size_keeps_most_frequent_tokens() -> None:
    """When limited, only the most frequent corpus tokens are kept."""
    vocabulary = build_vocabulary([["a", "a", "a"], ["b", "b"], ["c"]], max_size=2)

    assert len(vocabulary) == 2
    assert "a" in vocabulary
    assert "b" in vocabulary
    assert "c" not in vocabulary


def test_max_size_includes_special_tokens() -> None:
    """The size limit counts special tokens against the total."""
    vocabulary = build_vocabulary([["a", "a"], ["b"]], unk_token="<unk>", max_size=2)

    assert len(vocabulary) == 2
    assert "<unk>" in vocabulary
    assert "a" in vocabulary
    assert "b" not in vocabulary


def test_max_size_smaller_than_specials_raises() -> None:
    """A size limit smaller than the special-token count is rejected."""
    with pytest.raises(ValueError, match="max_size"):
        build_vocabulary([["a"]], unk_token="<unk>", pad_token="<pad>", max_size=1)


def test_negative_max_size_raises() -> None:
    """A negative size limit is rejected."""
    with pytest.raises(ValueError, match="max_size"):
        build_vocabulary([["a"]], max_size=-1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_corpus_yields_empty_vocabulary() -> None:
    """Building from an empty corpus yields an empty vocabulary."""
    vocabulary = build_vocabulary([])

    assert len(vocabulary) == 0


def test_empty_corpus_with_specials_contains_only_specials() -> None:
    """With no corpus tokens, only the special tokens remain."""
    vocabulary = build_vocabulary([], unk_token="<unk>", pad_token="<pad>")

    assert set(vocabulary.token_to_id) == {"<unk>", "<pad>"}
    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1


def test_builder_is_deterministic_across_calls() -> None:
    """The same corpus always produces an equal vocabulary."""
    corpus = [["the", "cat", "sat"], ["the", "dog", "sat"]]

    assert build_vocabulary(corpus) == build_vocabulary(corpus)
