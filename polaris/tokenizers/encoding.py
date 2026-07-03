"""Immutable representation of tokenizer output.

Design Principles
------------------
- Encoding is a pure value object: it holds data produced by a tokenizer
  and nothing else. It contains no tokenization logic, no vocabulary
  lookups, and no serialization behavior.
- Immutability is enforced via a frozen, slotted dataclass so that
  instances are safe to share, cache, and hash.
- Equality and hashing are derived directly from the dataclass fields,
  giving value semantics: two ``Encoding`` instances with the same
  ``ids`` and ``tokens`` compare equal.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Encoding"]


@dataclass(frozen=True, slots=True)
class Encoding:
    """Immutable output of a tokenizer applied to a piece of text.

    An ``Encoding`` pairs the integer ids produced by a tokenizer with
    the string tokens they correspond to. It carries no knowledge of how
    the ids or tokens were produced, and no knowledge of vocabularies,
    attention masks, offsets, or token type ids.

    Parameters
    ----------
    ids : tuple[int, ...]
        The sequence of integer token ids.
    tokens : tuple[str, ...]
        The sequence of string tokens corresponding positionally to
        ``ids``.

    Raises
    ------
    ValueError
        If ``ids`` and ``tokens`` do not have the same length.

    Examples
    --------
    >>> encoding = Encoding(ids=(1, 2, 3), tokens=("hello", "world", "!"))
    >>> len(encoding)
    3
    >>> encoding.ids
    (1, 2, 3)
    >>> encoding.tokens
    ('hello', 'world', '!')
    """

    ids: tuple[int, ...]
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that ``ids`` and ``tokens`` are aligned.

        Raises
        ------
        ValueError
            If ``ids`` and ``tokens`` do not have the same length.
        """
        if len(self.ids) != len(self.tokens):
            msg = (
                "ids and tokens must have the same length, got "
                f"{len(self.ids)} ids and {len(self.tokens)} tokens"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of ids (equivalently, tokens) in the encoding.

        Returns
        -------
        int
            The number of tokens in this encoding.

        Examples
        --------
        >>> len(Encoding(ids=(1, 2), tokens=("a", "b")))
        2
        """
        return len(self.ids)
