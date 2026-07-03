"""Unit tests freezing the public contract of the ``Tokenizer`` protocol.

These tests do not exercise any real tokenization algorithm. Instead, a
minimal fake implementation is defined locally to prove that an object
satisfying the :class:`~polaris.tokenizers.tokenizer.Tokenizer` shape is
recognized as such via ``@runtime_checkable``, and that each protocol
member returns the type specified by the contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from polaris.tokenizers.encoding import Encoding
from polaris.tokenizers.tokenizer import Tokenizer
from polaris.tokenizers.vocabulary import Vocabulary


class FakeTokenizer:
    """Minimal stand-in satisfying the `Tokenizer` protocol.

    Exists solely to exercise the protocol's public contract. Behavior is
    intentionally trivial: text is split on whitespace, ids are assigned
    via a fixed vocabulary, and decoding simply rejoins looked-up tokens
    with spaces.
    """

    def __init__(self) -> None:
        self._vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.split())

    def encode(self, text: str) -> Encoding:
        tokens = self.tokenize(text)
        ids = tuple(self._vocabulary.lookup_id(token) for token in tokens)
        return Encoding(ids=ids, tokens=tokens)

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._vocabulary.lookup_token(i) for i in ids)


def test_fake_tokenizer_satisfies_protocol() -> None:
    fake: Tokenizer = FakeTokenizer()

    assert isinstance(fake, Tokenizer)


def test_non_conforming_object_not_protocol() -> None:
    not_a_tokenizer = object()

    assert not isinstance(not_a_tokenizer, Tokenizer)


def test_vocabulary_returns_vocabulary() -> None:
    fake = FakeTokenizer()

    assert isinstance(fake.vocabulary, Vocabulary)


def test_tokenize_returns_tuple() -> None:
    fake = FakeTokenizer()

    result = fake.tokenize("hello world")

    assert isinstance(result, tuple)
    assert all(isinstance(token, str) for token in result)


def test_encode_returns_encoding() -> None:
    fake = FakeTokenizer()

    result = fake.encode("hello world")

    assert isinstance(result, Encoding)


def test_encode_alignment() -> None:
    fake = FakeTokenizer()

    result = fake.encode("hello world")

    assert len(result.ids) == len(result.tokens)


def test_decode_returns_string() -> None:
    fake = FakeTokenizer()
    encoding = fake.encode("hello world")

    result = fake.decode(encoding.ids)

    assert isinstance(result, str)
