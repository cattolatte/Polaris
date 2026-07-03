"""Unit tests for :class:`polaris.tokenizers.vocabulary.Vocabulary`.

Design Principles
------------------
- Tests exercise only the public contract of ``Vocabulary``:
  construction, validation, lookup, properties, membership,
  immutability, equality, and representation.
- No mocks or helper classes are used since ``Vocabulary`` has no
  collaborators.
- All tests run fully offline with no network access.
"""

from __future__ import annotations

import dataclasses

import pytest

from polaris.tokenizers.vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_with_multiple_tokens() -> None:
    """A ``Vocabulary`` can be constructed from a valid token-to-id mapping."""
    vocabulary = Vocabulary({"hello": 0, "world": 1, "!": 2})

    assert len(vocabulary) == 3
    assert vocabulary.lookup_id("hello") == 0
    assert vocabulary.lookup_id("world") == 1
    assert vocabulary.lookup_id("!") == 2


def test_construction_with_empty_vocabulary() -> None:
    """A ``Vocabulary`` can be constructed with an empty mapping."""
    vocabulary = Vocabulary({})

    assert len(vocabulary) == 0
    assert vocabulary.size == 0


def test_construction_with_single_token() -> None:
    """A ``Vocabulary`` can be constructed with exactly one token."""
    vocabulary = Vocabulary({"only": 0})

    assert len(vocabulary) == 1
    assert vocabulary.lookup_id("only") == 0
    assert vocabulary.lookup_token(0) == "only"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_duplicate_ids_raise_value_error() -> None:
    """Constructing with duplicate ids raises ``ValueError``."""
    with pytest.raises(ValueError, match="unique"):
        Vocabulary({"hello": 0, "world": 0})


def test_negative_ids_raise_value_error() -> None:
    """Constructing with a negative id raises ``ValueError``."""
    with pytest.raises(ValueError, match="non-negative"):
        Vocabulary({"hello": -1, "world": 0})


def test_non_contiguous_ids_raise_value_error() -> None:
    """Constructing with non-contiguous ids raises ``ValueError``."""
    with pytest.raises(ValueError, match="contiguous"):
        Vocabulary({"hello": 0, "world": 2})


def test_non_contiguous_ids_not_starting_at_zero_raise_value_error() -> None:
    """Ids that skip zero entirely are rejected as non-contiguous."""
    with pytest.raises(ValueError, match="contiguous"):
        Vocabulary({"hello": 1, "world": 2})


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_lookup_id_returns_correct_id() -> None:
    """``lookup_id`` returns the id assigned to a known token."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert vocabulary.lookup_id("world") == 1


def test_lookup_token_returns_correct_token() -> None:
    """``lookup_token`` returns the token assigned to a known id."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert vocabulary.lookup_token(0) == "hello"


def test_lookup_id_with_unknown_token_raises_key_error() -> None:
    """``lookup_id`` raises ``KeyError`` for an unknown token."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(KeyError):
        vocabulary.lookup_id("cat")


def test_lookup_token_with_unknown_id_raises_key_error() -> None:
    """``lookup_token`` raises ``KeyError`` for an unknown id."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(KeyError):
        vocabulary.lookup_token(99)


def test_lookup_round_trip() -> None:
    """Resolving a token to an id and back returns the original token."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    for token in vocabulary.token_to_id:
        assert vocabulary.lookup_token(vocabulary.lookup_id(token)) == token


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_size_matches_length() -> None:
    """The ``size`` property agrees with ``len(vocabulary)``."""
    vocabulary = Vocabulary({"hello": 0, "world": 1, "!": 2})

    assert vocabulary.size == 3
    assert len(vocabulary) == 3


def test_token_to_id_mapping_contents() -> None:
    """The ``token_to_id`` mapping reflects the constructor argument."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert dict(vocabulary.token_to_id) == {"hello": 0, "world": 1}


def test_id_to_token_mapping_contents() -> None:
    """The ``id_to_token`` mapping is the reverse of ``token_to_id``."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert dict(vocabulary.id_to_token) == {0: "hello", 1: "world"}


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def test_membership_with_existing_token() -> None:
    """An existing token is reported as a member of the vocabulary."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert "hello" in vocabulary


def test_membership_with_missing_token() -> None:
    """A token absent from the vocabulary is not reported as a member."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert "cat" not in vocabulary


def test_membership_with_non_string_object_returns_false() -> None:
    """A non-string object is never a member of the vocabulary."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert 0 not in vocabulary
    assert None not in vocabulary


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_token_to_id_field_cannot_be_reassigned() -> None:
    """Reassigning the ``token_to_id`` field raises ``FrozenInstanceError``."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(dataclasses.FrozenInstanceError):
        vocabulary.token_to_id = {"new": 0}  # type: ignore[misc]


def test_id_to_token_field_cannot_be_reassigned() -> None:
    """Reassigning the ``id_to_token`` field raises ``FrozenInstanceError``."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(dataclasses.FrozenInstanceError):
        vocabulary.id_to_token = {0: "new"}  # type: ignore[misc]


def test_token_to_id_mapping_cannot_be_modified() -> None:
    """The ``token_to_id`` mapping rejects item assignment."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(TypeError):
        vocabulary.token_to_id["new"] = 2  # type: ignore[index]


def test_id_to_token_mapping_cannot_be_modified() -> None:
    """The ``id_to_token`` mapping rejects item assignment."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    with pytest.raises(TypeError):
        vocabulary.id_to_token[2] = "new"  # type: ignore[index]


def test_vocabulary_is_a_frozen_dataclass() -> None:
    """``Vocabulary`` is declared as a dataclass."""
    assert dataclasses.is_dataclass(Vocabulary)


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_equal_vocabularies_compare_equal() -> None:
    """Two vocabularies with identical mappings compare equal."""
    first = Vocabulary({"hello": 0, "world": 1})
    second = Vocabulary({"hello": 0, "world": 1})

    assert first == second


def test_different_vocabularies_compare_unequal() -> None:
    """Vocabularies with different mappings compare unequal."""
    first = Vocabulary({"hello": 0, "world": 1})
    second = Vocabulary({"hello": 0, "cat": 1})

    assert first != second


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_repr_contains_class_name() -> None:
    """The repr of a ``Vocabulary`` includes the class name."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert "Vocabulary" in repr(vocabulary)


def test_repr_contains_token_mapping() -> None:
    """The repr of a ``Vocabulary`` includes its token-to-id mapping."""
    vocabulary = Vocabulary({"hello": 0, "world": 1})

    assert "hello" in repr(vocabulary)
    assert "world" in repr(vocabulary)
