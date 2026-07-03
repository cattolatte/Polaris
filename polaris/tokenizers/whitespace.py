"""Whitespace-based tokenizer implementation for Polaris.

This module provides ``WhitespaceTokenizer``, the first concrete
implementation of the :class:`~polaris.tokenizers.tokenizer.Tokenizer`
protocol. Its purpose is to prove that the tokenizer architecture works
end-to-end using the simplest possible strategy: splitting text on
whitespace via Python's built-in :meth:`str.split`.

It intentionally performs no preprocessing, normalization, punctuation
handling, casing changes, padding, or truncation. Unknown-token handling is
delegated to the vocabulary: if the vocabulary defines an ``unk_token``,
:meth:`WhitespaceTokenizer.encode` maps out-of-vocabulary tokens to its id;
otherwise encoding an unknown token raises ``KeyError``.
"""

from __future__ import annotations

from collections.abc import Sequence

from polaris.tokenizers.encoding import Encoding
from polaris.tokenizers.vocabulary import Vocabulary

__all__ = ["WhitespaceTokenizer"]


class WhitespaceTokenizer:
    """A tokenizer that splits text on whitespace.

    `WhitespaceTokenizer` is a minimal, dependency-free reference
    implementation of the :class:`~polaris.tokenizers.tokenizer.Tokenizer`
    protocol. It splits raw text using Python's built-in
    :meth:`str.split` and relies entirely on a supplied
    :class:`~polaris.tokenizers.vocabulary.Vocabulary` for id lookups.

    Parameters
    ----------
    vocabulary : Vocabulary
        The vocabulary used to convert between tokens and ids.

    Examples
    --------
    >>> from polaris.tokenizers.vocabulary import Vocabulary
    >>> from polaris.tokenizers.whitespace import WhitespaceTokenizer
    >>> vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
    >>> tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
    >>> tokenizer.tokenize("hello world")
    ('hello', 'world')
    >>> encoding = tokenizer.encode("hello world")
    >>> encoding.ids
    (0, 1)
    >>> tokenizer.decode((0, 1))
    'hello world'
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> Vocabulary:
        """Vocabulary: The vocabulary used by this tokenizer.

        Returns
        -------
        Vocabulary
            The immutable vocabulary supplied at construction time.

        Examples
        --------
        >>> from polaris.tokenizers.vocabulary import Vocabulary
        >>> from polaris.tokenizers.whitespace import WhitespaceTokenizer
        >>> vocabulary = Vocabulary(token_to_id={"hello": 0})
        >>> tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
        >>> tokenizer.vocabulary is vocabulary
        True
        """
        return self._vocabulary

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split raw text into tokens on whitespace.

        Parameters
        ----------
        text : str
            The raw input text to tokenize.

        Returns
        -------
        tuple[str, ...]
            The tokens produced by splitting `text` on whitespace.

        Examples
        --------
        >>> from polaris.tokenizers.vocabulary import Vocabulary
        >>> from polaris.tokenizers.whitespace import WhitespaceTokenizer
        >>> vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
        >>> tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
        >>> tokenizer.tokenize("hello world")
        ('hello', 'world')
        """
        return tuple(text.split())

    def encode(self, text: str) -> Encoding:
        """Tokenize text and convert it into a complete `Encoding`.

        Parameters
        ----------
        text : str
            The raw input text to encode.

        Returns
        -------
        Encoding
            An immutable structure containing the aligned token ids and
            token strings produced from `text`.

        Raises
        ------
        KeyError
            If a token produced by :meth:`tokenize` is not present in the
            vocabulary and the vocabulary defines no ``unk_token``. When an
            ``unk_token`` is configured, unknown tokens map to its id instead.

        Examples
        --------
        >>> from polaris.tokenizers.vocabulary import Vocabulary
        >>> from polaris.tokenizers.whitespace import WhitespaceTokenizer
        >>> vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
        >>> tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
        >>> tokenizer.encode("hello world")
        Encoding(ids=(0, 1), tokens=('hello', 'world'))
        """
        tokens = self.tokenize(text)
        ids = tuple(self._vocabulary.get_id(token) for token in tokens)
        return Encoding(ids=ids, tokens=tokens)

    def decode(self, ids: Sequence[int]) -> str:
        """Convert a sequence of token ids back into text.

        Parameters
        ----------
        ids : Sequence[int]
            The token ids to convert back into text.

        Returns
        -------
        str
            The tokens corresponding to `ids`, looked up in the
            tokenizer's vocabulary and joined with a single space.

        Raises
        ------
        KeyError
            If any id in `ids` is not present in the tokenizer's
            vocabulary.

        Examples
        --------
        >>> from polaris.tokenizers.vocabulary import Vocabulary
        >>> from polaris.tokenizers.whitespace import WhitespaceTokenizer
        >>> vocabulary = Vocabulary(token_to_id={"hello": 0, "world": 1})
        >>> tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
        >>> tokenizer.decode((0, 1))
        'hello world'
        """
        tokens = (self._vocabulary.lookup_token(id_) for id_ in ids)
        return " ".join(tokens)
