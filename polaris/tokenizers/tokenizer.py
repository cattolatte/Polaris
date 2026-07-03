"""Tokenizer protocol definition for Polaris.

This module defines the common behavioral contract that every tokenizer
implementation within Polaris must satisfy. It contains no tokenization
logic itself; it exists purely to describe the public interface shared by
all concrete tokenizers (e.g. ``WhitespaceTokenizer``, ``BPETokenizer``,
``WordPieceTokenizer``).

The protocol is intentionally minimal and stateless from the caller's
perspective: it exposes only the ability to tokenize raw text, encode text
into an immutable :class:`~polaris.tokenizers.encoding.Encoding`, decode a
sequence of ids back into text, and access the tokenizer's
:class:`~polaris.tokenizers.vocabulary.Vocabulary`.

Examples
--------
>>> from polaris.tokenizers.tokenizer import Tokenizer
>>> def uses_any_tokenizer(tokenizer: Tokenizer, text: str) -> None:
...     encoding = tokenizer.encode(text)
...     tokens = tokenizer.tokenize(text)
...     restored = tokenizer.decode(encoding.ids)
...     vocab_size = len(tokenizer.vocabulary)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from polaris.tokenizers.encoding import Encoding
from polaris.tokenizers.vocabulary import Vocabulary

__all__ = ["Tokenizer"]


@runtime_checkable
class Tokenizer(Protocol):
    """Behavioral contract shared by all Polaris tokenizer implementations.

    This protocol defines the public surface that every tokenizer must
    expose: converting raw text into tokens, encoding text into an
    :class:`~polaris.tokenizers.encoding.Encoding`, decoding ids back into
    text, and exposing the underlying
    :class:`~polaris.tokenizers.vocabulary.Vocabulary`.

    `Tokenizer` performs no tokenization itself and carries no
    implementation details. It is a structural interface: any object that
    satisfies this shape may be treated as a `Tokenizer`, regardless of
    inheritance.

    Notes
    -----
    Concrete implementations may internally call :meth:`tokenize` from
    within :meth:`encode`, but this relationship is an implementation
    detail and is not specified or required by this protocol.

    Examples
    --------
    >>> from polaris.tokenizers.tokenizer import Tokenizer
    >>> def describe(tokenizer: Tokenizer) -> str:
    ...     return f"vocabulary size: {len(tokenizer.vocabulary)}"
    """

    @property
    def vocabulary(self) -> Vocabulary:
        """Vocabulary: The vocabulary used by this tokenizer.

        Returns
        -------
        Vocabulary
            The immutable, bidirectional token-to-id mapping associated
            with this tokenizer.

        Examples
        --------
        >>> from polaris.tokenizers.tokenizer import Tokenizer
        >>> def vocab_size(tokenizer: Tokenizer) -> int:
        ...     return len(tokenizer.vocabulary)
        """
        ...

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split raw text into a sequence of string tokens.

        Parameters
        ----------
        text : str
            The raw input text to tokenize.

        Returns
        -------
        tuple[str, ...]
            The ordered tokens produced from `text`. No id lookup or
            vocabulary interaction is performed.

        Examples
        --------
        >>> from polaris.tokenizers.tokenizer import Tokenizer
        >>> def first_token(tokenizer: Tokenizer, text: str) -> str:
        ...     return tokenizer.tokenize(text)[0]
        """
        ...

    def encode(self, text: str) -> Encoding:
        """Convert raw text into a complete `Encoding`.

        Parameters
        ----------
        text : str
            The raw input text to encode.

        Returns
        -------
        Encoding
            An immutable structure containing the aligned token ids and
            token strings produced from `text`.

        Examples
        --------
        >>> from polaris.tokenizers.tokenizer import Tokenizer
        >>> def encode_text(tokenizer: Tokenizer, text: str) -> Encoding:
        ...     return tokenizer.encode(text)
        """
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Convert a sequence of token ids back into text.

        Parameters
        ----------
        ids : Sequence[int]
            The token ids to convert back into text.

        Returns
        -------
        str
            The text reconstructed from `ids`.

        Examples
        --------
        >>> from polaris.tokenizers.tokenizer import Tokenizer
        >>> def decode_ids(tokenizer: Tokenizer, ids: Sequence[int]) -> str:
        ...     return tokenizer.decode(ids)
        """
        ...
