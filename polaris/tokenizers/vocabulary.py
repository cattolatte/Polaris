"""Immutable mapping between string tokens and integer token ids.

Design Principles
------------------
- ``Vocabulary`` is a pure value object: it represents a validated,
  bidirectional mapping between tokens and ids, and nothing else.
- It contains no tokenizer logic, no training logic, no serialization,
  no frequency information, and no pruning.
- Immutability is enforced the same way as :class:`~polaris.tokenizers.
  encoding.Encoding`: a frozen, slotted dataclass whose fields are
  stored as read-only ``MappingProxyType`` views, with equality and
  hashing derived by the dataclass machinery rather than hand-written.
- The reverse mapping (``id`` to ``token``) is always derived
  automatically from the constructor argument in ``__post_init__``; it
  is never supplied or constructed manually by callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ["Vocabulary"]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """An immutable, validated mapping between tokens and integer ids.

    A ``Vocabulary`` wraps a token-to-id mapping, validates its
    invariants once at construction time, and derives the reverse
    id-to-token mapping automatically.

    Parameters
    ----------
    token_to_id : Mapping[str, int]
        A mapping from token strings to integer ids. Tokens are unique
        by construction of the mapping itself; ids must additionally be
        unique, non-negative, and contiguous starting from ``0``.

    Raises
    ------
    ValueError
        If any token id is negative, if token ids are not unique, or
        if the ids are not contiguous starting from ``0``.

    Examples
    --------
    >>> vocabulary = Vocabulary({"hello": 0, "world": 1})
    >>> len(vocabulary)
    2
    >>> "hello" in vocabulary
    True
    >>> "cat" in vocabulary
    False
    >>> vocabulary.lookup_id("world")
    1
    >>> vocabulary.lookup_token(0)
    """

    token_to_id: Mapping[str, int]
    id_to_token: Mapping[int, str] = field(init=False)

    def __post_init__(self) -> None:
        """Validate invariants and derive the reverse mapping.

        Raises
        ------
        ValueError
            If any token id is negative, if token ids are not unique,
            or if the ids are not contiguous starting from ``0``.
        """
        ids = list(self.token_to_id.values())

        if len(set(ids)) != len(ids):
            msg = "token ids must be unique"
            raise ValueError(msg)

        if any(token_id < 0 for token_id in ids):
            msg = "token ids must be non-negative"
            raise ValueError(msg)

        if sorted(ids) != list(range(len(ids))):
            msg = (
                "token ids must be contiguous starting from 0, got " f"{sorted(ids)!r}"
            )
            raise ValueError(msg)

        resolved_token_to_id = MappingProxyType(dict(self.token_to_id))
        resolved_id_to_token = MappingProxyType(
            {token_id: token for token, token_id in resolved_token_to_id.items()}
        )

        object.__setattr__(self, "token_to_id", resolved_token_to_id)
        object.__setattr__(self, "id_to_token", resolved_id_to_token)

    @property
    def size(self) -> int:
        """int: The number of tokens in the vocabulary.

        Examples
        --------
        >>> Vocabulary({"hello": 0, "world": 1}).size
        2
        """
        return len(self.token_to_id)

    def lookup_id(self, token: str) -> int:
        """Look up the integer id assigned to a token.

        Parameters
        ----------
        token : str
            The token to resolve.

        Returns
        -------
        int
            The id assigned to ``token``.

        Raises
        ------
        KeyError
            If ``token`` is not present in the vocabulary.

        Examples
        --------
        >>> Vocabulary({"hello": 0, "world": 1}).lookup_id("world")
        1
        """
        try:
            return self.token_to_id[token]
        except KeyError:
            msg = f"unknown token: {token!r}"
            raise KeyError(msg) from None

    def lookup_token(self, token_id: int) -> str:
        """Look up the token assigned to an integer id.

        Parameters
        ----------
        token_id : int
            The id to resolve.

        Returns
        -------
        str
            The token assigned to ``token_id``.

        Raises
        ------
        KeyError
            If ``token_id`` is not present in the vocabulary.

        Examples
        --------
        >>> Vocabulary({"hello": 0, "world": 1}).lookup_token(0)
        'hello'
        """
        try:
            return self.id_to_token[token_id]
        except KeyError:
            msg = f"unknown id: {token_id!r}"
            raise KeyError(msg) from None

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.

        Examples
        --------
        >>> len(Vocabulary({"hello": 0, "world": 1}))
        2
        """
        return len(self.token_to_id)

    def __contains__(self, token: object) -> bool:
        """Return whether a token exists in the vocabulary.

        Parameters
        ----------
        token : object
            The value to check for membership. Only string tokens can
            be present; any other type returns ``False``.

        Returns
        -------
        bool
            ``True`` if ``token`` is a token in the vocabulary,
            ``False`` otherwise.

        Examples
        --------
        >>> "hello" in Vocabulary({"hello": 0, "world": 1})
        True
        >>> "cat" in Vocabulary({"hello": 0, "world": 1})
        False
        """
        return token in self.token_to_id
