"""Immutable mapping between string tokens and integer token ids.

Design Principles
------------------
- ``Vocabulary`` is a value object: it represents a validated, bidirectional
  mapping between tokens and ids, plus the identity of its special tokens
  (unknown and padding), and nothing else.
- It contains no *construction* logic: it does not count frequencies, prune,
  or build itself from a corpus. That lives in
  :func:`~polaris.tokenizers.vocabulary_builder.build_vocabulary`, keeping this
  type a pure, deterministic value object.
- Immutability is enforced the same way as :class:`~polaris.tokenizers.
  encoding.Encoding`: a frozen, slotted dataclass whose fields are
  stored as read-only ``MappingProxyType`` views, with equality and
  hashing derived by the dataclass machinery rather than hand-written.
- The reverse mapping (``id`` to ``token``) and the special-token ids are
  always derived automatically in ``__post_init__``; they are never supplied
  or constructed manually by callers.
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
    invariants once at construction time, derives the reverse
    id-to-token mapping automatically, and optionally designates an
    unknown-token and a padding-token.

    Parameters
    ----------
    token_to_id : Mapping[str, int]
        A mapping from token strings to integer ids. Tokens are unique
        by construction of the mapping itself; ids must additionally be
        unique, non-negative, and contiguous starting from ``0``.
    unk_token : str, optional
        The token used to represent out-of-vocabulary tokens. If given, it
        must be a key of ``token_to_id`` and enables the fallback behaviour
        of :meth:`get_id`.
    pad_token : str, optional
        The token used to pad sequences to equal length. If given, it must
        be a key of ``token_to_id``. ``Vocabulary`` only records its
        identity; padding itself is performed by the collation layer.
    mask_token : str, optional
        The token substituted for hidden positions during masked-language-model
        pretraining. If given, it must be a key of ``token_to_id``.
        ``Vocabulary`` only records its identity; masking itself is performed by
        the pretraining layer.

    Raises
    ------
    ValueError
        If any token id is negative, if token ids are not unique, if the
        ids are not contiguous starting from ``0``, if a supplied special
        token is not present in ``token_to_id``, or if any two configured
        special tokens are equal.

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
    'hello'
    >>> vocab = Vocabulary(
    ...     {"<pad>": 0, "<unk>": 1, "hello": 2},
    ...     unk_token="<unk>",
    ...     pad_token="<pad>",
    ... )
    >>> vocab.get_id("unseen")
    1
    """

    token_to_id: Mapping[str, int]
    unk_token: str | None = None
    pad_token: str | None = None
    mask_token: str | None = None

    id_to_token: Mapping[int, str] = field(init=False)
    unk_id: int | None = field(init=False)
    pad_id: int | None = field(init=False)
    mask_id: int | None = field(init=False)

    def __post_init__(self) -> None:
        """Validate invariants and derive the reverse mapping and special ids.

        Raises
        ------
        ValueError
            If any token id is negative, if token ids are not unique, if the
            ids are not contiguous starting from ``0``, if a supplied special
            token is absent, or if ``unk_token`` and ``pad_token`` are equal.
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

        specials = (
            ("unk_token", self.unk_token),
            ("pad_token", self.pad_token),
            ("mask_token", self.mask_token),
        )
        for role, token in specials:
            if token is not None and token not in resolved_token_to_id:
                msg = f"{role} {token!r} is not present in the vocabulary"
                raise ValueError(msg)

        seen_specials: dict[str, str] = {}
        for role, token in specials:
            if token is None:
                continue
            if token in seen_specials:
                msg = (
                    f"{role} and {seen_specials[token]} must be different tokens, "
                    f"both are {token!r}"
                )
                raise ValueError(msg)
            seen_specials[token] = role

        object.__setattr__(self, "token_to_id", resolved_token_to_id)
        object.__setattr__(self, "id_to_token", resolved_id_to_token)
        object.__setattr__(
            self,
            "unk_id",
            None if self.unk_token is None else resolved_token_to_id[self.unk_token],
        )
        object.__setattr__(
            self,
            "pad_id",
            None if self.pad_token is None else resolved_token_to_id[self.pad_token],
        )
        object.__setattr__(
            self,
            "mask_id",
            None if self.mask_token is None else resolved_token_to_id[self.mask_token],
        )

    @property
    def size(self) -> int:
        """int: The number of tokens in the vocabulary.

        Examples
        --------
        >>> Vocabulary({"hello": 0, "world": 1}).size
        2
        """
        return len(self.token_to_id)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        """tuple[str, ...]: The configured special tokens, padding first.

        Contains whichever of ``pad_token``, ``unk_token`` and ``mask_token``
        are set, in that order. Empty when none is configured.

        Examples
        --------
        >>> vocab = Vocabulary(
        ...     {"<pad>": 0, "<unk>": 1}, unk_token="<unk>", pad_token="<pad>"
        ... )
        >>> vocab.special_tokens
        ('<pad>', '<unk>')
        """
        return tuple(
            token
            for token in (self.pad_token, self.unk_token, self.mask_token)
            if token is not None
        )

    def lookup_id(self, token: str) -> int:
        """Look up the integer id assigned to a token, strictly.

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
            If ``token`` is not present in the vocabulary. Unlike
            :meth:`get_id`, this never falls back to the unknown-token id.

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

    def get_id(self, token: str) -> int:
        """Resolve a token to an id, using the unknown-token fallback.

        Parameters
        ----------
        token : str
            The token to resolve.

        Returns
        -------
        int
            The id assigned to ``token``; or the ``unk_token`` id if ``token``
            is absent and an ``unk_token`` is configured.

        Raises
        ------
        KeyError
            If ``token`` is absent and no ``unk_token`` is configured.

        Examples
        --------
        >>> vocab = Vocabulary({"<unk>": 0, "hello": 1}, unk_token="<unk>")
        >>> vocab.get_id("hello")
        1
        >>> vocab.get_id("missing")
        0
        """
        token_id = self.token_to_id.get(token)
        if token_id is not None:
            return token_id
        if self.unk_id is not None:
            return self.unk_id
        msg = f"unknown token: {token!r}"
        raise KeyError(msg)

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
