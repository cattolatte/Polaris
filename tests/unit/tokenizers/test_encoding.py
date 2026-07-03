"""Unit tests for :class:`polaris.tokenizers.encoding.Encoding`.

Design Principles
------------------
- Tests exercise only the public contract of ``Encoding``: construction,
  validation, immutability, equality, hashing, length, properties, and
  representation.
- No mocks are used since ``Encoding`` has no collaborators.
- All tests run fully offline with no network access.
"""

from __future__ import annotations

import dataclasses

import pytest

from polaris.tokenizers.encoding import Encoding

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_with_multiple_tokens() -> None:
    """An ``Encoding`` can be constructed from aligned ids and tokens."""
    ids = (1, 2, 3, 4, 5)
    tokens = ("a", "b", "c", "d", "e")

    encoding = Encoding(ids=ids, tokens=tokens)

    assert encoding.ids == ids
    assert encoding.tokens == tokens
    assert len(encoding) == 5


def test_construction_with_empty_encoding() -> None:
    """An ``Encoding`` can be constructed with empty ids and tokens."""
    encoding = Encoding(ids=(), tokens=())

    assert encoding.ids == ()
    assert encoding.tokens == ()
    assert len(encoding) == 0


def test_construction_with_single_token() -> None:
    """An ``Encoding`` can be constructed with exactly one id and token."""
    encoding = Encoding(ids=(42,), tokens=("token",))

    assert encoding.ids == (42,)
    assert encoding.tokens == ("token",)
    assert len(encoding) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise_value_error() -> None:
    """Constructing with mismatched ids and tokens raises ``ValueError``."""
    with pytest.raises(ValueError):
        Encoding(ids=(1, 2, 3), tokens=("only", "two"))


def test_mismatched_lengths_error_message_mentions_lengths() -> None:
    """The ``ValueError`` message reports the mismatched lengths."""
    with pytest.raises(ValueError, match=r"\b3\b.*\b2\b|\b2\b.*\b3\b"):
        Encoding(ids=(1, 2, 3), tokens=("only", "two"))


def test_mismatched_lengths_ids_longer_than_tokens() -> None:
    """A ``ValueError`` is raised when ``ids`` is longer than ``tokens``."""
    with pytest.raises(ValueError):
        Encoding(ids=(1, 2), tokens=("only_one",))


def test_mismatched_lengths_tokens_longer_than_ids() -> None:
    """A ``ValueError`` is raised when ``tokens`` is longer than ``ids``."""
    with pytest.raises(ValueError):
        Encoding(ids=(1,), tokens=("one", "two"))


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_ids_field_cannot_be_reassigned() -> None:
    """Reassigning the ``ids`` field raises ``FrozenInstanceError``."""
    encoding = Encoding(ids=(1, 2), tokens=("a", "b"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.ids = (3, 4)  # type: ignore[misc]


def test_tokens_field_cannot_be_reassigned() -> None:
    """Reassigning the ``tokens`` field raises ``FrozenInstanceError``."""
    encoding = Encoding(ids=(1, 2), tokens=("a", "b"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.tokens = ("c", "d")  # type: ignore[misc]


def test_encoding_is_a_frozen_dataclass() -> None:
    """``Encoding`` is declared as a dataclass."""
    assert dataclasses.is_dataclass(Encoding)


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_equal_encodings_compare_equal() -> None:
    """Two encodings with identical ids and tokens compare equal."""
    first = Encoding(ids=(1, 2), tokens=("a", "b"))
    second = Encoding(ids=(1, 2), tokens=("a", "b"))

    assert first == second


def test_different_ids_compare_unequal() -> None:
    """Encodings with different ids compare unequal."""
    first = Encoding(ids=(1, 2), tokens=("a", "b"))
    second = Encoding(ids=(1, 3), tokens=("a", "b"))

    assert first != second


def test_different_tokens_compare_unequal() -> None:
    """Encodings with different tokens compare unequal."""
    first = Encoding(ids=(1, 2), tokens=("a", "b"))
    second = Encoding(ids=(1, 2), tokens=("a", "c"))

    assert first != second


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_encoding_is_hashable() -> None:
    """An ``Encoding`` instance can be hashed."""
    encoding = Encoding(ids=(1, 2), tokens=("a", "b"))

    assert isinstance(hash(encoding), int)


def test_equal_encodings_produce_equal_hashes() -> None:
    """Equal encodings hash to the same value."""
    first = Encoding(ids=(1, 2), tokens=("a", "b"))
    second = Encoding(ids=(1, 2), tokens=("a", "b"))

    assert hash(first) == hash(second)


def test_encoding_can_be_used_as_a_set_member() -> None:
    """Hashability allows an ``Encoding`` to be stored in a set."""
    first = Encoding(ids=(1, 2), tokens=("a", "b"))
    second = Encoding(ids=(1, 2), tokens=("a", "b"))

    assert {first, second} == {first}


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


def test_len_returns_number_of_ids() -> None:
    """``__len__`` returns the number of ids in the encoding."""
    encoding = Encoding(ids=(1, 2, 3), tokens=("a", "b", "c"))

    assert len(encoding) == len(encoding.ids)


def test_len_equals_number_of_tokens() -> None:
    """``__len__`` also equals the number of tokens in the encoding."""
    encoding = Encoding(ids=(1, 2, 3), tokens=("a", "b", "c"))

    assert len(encoding) == len(encoding.tokens)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_ids_are_preserved_exactly() -> None:
    """The ``ids`` property returns values equal to what was passed in."""
    ids = (7, 8, 9)

    encoding = Encoding(ids=ids, tokens=("x", "y", "z"))

    assert encoding.ids == ids


def test_tokens_are_preserved_exactly() -> None:
    """The ``tokens`` property returns values equal to what was passed in."""
    tokens = ("x", "y", "z")

    encoding = Encoding(ids=(7, 8, 9), tokens=tokens)

    assert encoding.tokens == tokens


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_repr() -> None:
    """The repr of an ``Encoding`` reflects its class name and fields."""
    encoding = Encoding(ids=(1, 2), tokens=("a", "b"))

    assert repr(encoding) == "Encoding(ids=(1, 2), tokens=('a', 'b'))"
