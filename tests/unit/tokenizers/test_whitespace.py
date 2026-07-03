"""Unit tests for the `WhitespaceTokenizer` concrete implementation.

These tests verify only the public contract of `WhitespaceTokenizer`:
construction, protocol conformance, tokenization, encoding, decoding,
round-tripping, and representation. No implementation details, private
attributes, or third-party internals are exercised.
"""

from __future__ import annotations

import pytest

from polaris.tokenizers.encoding import Encoding
from polaris.tokenizers.tokenizer import Tokenizer
from polaris.tokenizers.vocabulary import Vocabulary
from polaris.tokenizers.whitespace import WhitespaceTokenizer

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_tokenizer_exposes_supplied_vocabulary() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.vocabulary == vocabulary


def test_vocabulary_property_returns_same_instance() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.vocabulary is vocabulary


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------


def test_whitespace_tokenizer_is_instance_of_tokenizer() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert isinstance(tokenizer, Tokenizer)


def test_whitespace_tokenizer_satisfies_typed_protocol_assignment() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer: Tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert isinstance(tokenizer, Tokenizer)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_empty_string_returns_empty_tuple() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("") == ()


def test_tokenize_single_word() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello") == ("hello",)


def test_tokenize_multiple_words() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello world") == ("hello", "world")


def test_tokenize_leading_whitespace() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("   hello") == ("hello",)


def test_tokenize_trailing_whitespace() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello   ") == ("hello",)


def test_tokenize_multiple_spaces_collapse() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello     world") == ("hello", "world")


def test_tokenize_tabs() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello\tworld") == ("hello", "world")


def test_tokenize_newlines() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello\nworld") == ("hello", "world")


def test_tokenize_mixed_whitespace() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("  hello \t\n world  ") == ("hello", "world")


def test_tokenize_preserves_punctuation() -> None:
    vocabulary = Vocabulary(token_to_id={"hello,": 0, "world!": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("hello, world!") == ("hello,", "world!")


def test_tokenize_preserves_casing() -> None:
    vocabulary = Vocabulary(token_to_id={"Hello": 0, "World": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.tokenize("Hello World") == ("Hello", "World")


# ---------------------------------------------------------------------------
# encode()
# ---------------------------------------------------------------------------


def test_encode_returns_encoding() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    result = tokenizer.encode("hello world")

    assert isinstance(result, Encoding)


def test_encode_ids_align_with_tokens() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    result = tokenizer.encode("hello world")

    assert len(result.ids) == len(result.tokens)


def test_encode_ids_match_vocabulary() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    result = tokenizer.encode("hello world")

    assert result.ids == (0, 1)
    assert result.tokens == ("hello", "world")


def test_encode_unknown_token_raises_key_error() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    with pytest.raises(KeyError):
        tokenizer.encode("hello world")


def test_encode_empty_text_returns_empty_encoding() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    result = tokenizer.encode("")

    assert result.ids == ()
    assert result.tokens == ()


# ---------------------------------------------------------------------------
# decode()
# ---------------------------------------------------------------------------


def test_decode_decodes_ids_correctly() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.decode((0, 1)) == "hello world"


def test_decode_empty_ids_returns_empty_string() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.decode(()) == ""


def test_decode_unknown_id_raises_key_error() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    with pytest.raises(KeyError):
        tokenizer.decode((99,))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_normalizes_whitespace() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    encoding = tokenizer.encode("  hello    world  ")

    assert tokenizer.decode(encoding.ids) == "hello world"


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_repr_contains_class_name() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert "WhitespaceTokenizer" in repr(tokenizer)


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


def test_len_of_tokenizer_vocabulary_works() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert len(tokenizer.vocabulary) == 2


def test_tokenizer_vocabulary_is_exactly_supplied_object() -> None:
    vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    assert tokenizer.vocabulary is vocabulary


# ---------------------------------------------------------------------------
# Unknown-token handling
# ---------------------------------------------------------------------------


def test_encode_maps_unknown_tokens_to_unk_id_when_configured() -> None:
    vocabulary = Vocabulary(token_to_id={"<unk>": 0, "hello": 1}, unk_token="<unk>")
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    encoding = tokenizer.encode("hello world")

    assert encoding.ids == (1, 0)
    assert encoding.tokens == ("hello", "world")


def test_encode_with_unk_configured_never_raises_on_unknown() -> None:
    vocabulary = Vocabulary(token_to_id={"<unk>": 0, "hello": 1}, unk_token="<unk>")
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    encoding = tokenizer.encode("totally unseen words")

    assert encoding.ids == (0, 0, 0)
