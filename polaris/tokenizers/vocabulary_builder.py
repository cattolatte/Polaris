"""Build a :class:`~polaris.tokenizers.vocabulary.Vocabulary` from a corpus.

Design Principles
------------------
- Construction logic (frequency counting, pruning, ordering) lives here,
  deliberately kept *out* of the ``Vocabulary`` value object so that
  ``Vocabulary`` stays a pure, validated mapping.
- ``build_vocabulary`` is a pure, deterministic function: the same inputs
  always produce the same vocabulary (ties are broken by token text), which
  supports Polaris' reproducibility goal.
- It consumes already-tokenized input (sequences of string tokens) and so
  has no dependency on any particular :class:`~polaris.tokenizers.tokenizer.
  Tokenizer`. Callers tokenize first, then build.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from polaris.tokenizers.vocabulary import Vocabulary

__all__ = ["build_vocabulary"]


def build_vocabulary(
    token_sequences: Iterable[Iterable[str]],
    *,
    unk_token: str | None = None,
    pad_token: str | None = None,
    min_frequency: int = 1,
    max_size: int | None = None,
) -> Vocabulary:
    """Build a ``Vocabulary`` from tokenized text.

    Special tokens are reserved first (padding before unknown), followed by
    corpus tokens ordered by descending frequency, with ties broken
    alphabetically for determinism. Ids are assigned contiguously from ``0``.

    Parameters
    ----------
    token_sequences : Iterable[Iterable[str]]
        The corpus as already-tokenized sequences (e.g. the output of a
        tokenizer's ``tokenize`` applied to each document).
    unk_token : str, optional
        If given, reserved as the unknown token and passed through to the
        resulting ``Vocabulary``.
    pad_token : str, optional
        If given, reserved as the padding token and passed through to the
        resulting ``Vocabulary``.
    min_frequency : int, default 1
        Corpus tokens appearing fewer than this many times are dropped. Must
        be at least ``1``.
    max_size : int, optional
        Maximum total vocabulary size, *including* special tokens. When set,
        only the most frequent corpus tokens are kept to fit the limit.

    Returns
    -------
    Vocabulary
        A vocabulary with contiguous ids and the given special tokens.

    Raises
    ------
    ValueError
        If ``min_frequency`` is less than ``1``, if ``max_size`` is negative,
        or if ``max_size`` is smaller than the number of special tokens.

    Examples
    --------
    >>> vocab = build_vocabulary(
    ...     [["good", "movie"], ["good", "film"]],
    ...     unk_token="<unk>",
    ...     pad_token="<pad>",
    ... )
    >>> vocab.pad_id, vocab.unk_id
    (0, 1)
    >>> vocab.lookup_id("good")
    2
    """
    if min_frequency < 1:
        msg = f"min_frequency must be at least 1, got {min_frequency}"
        raise ValueError(msg)

    specials: list[str] = []
    for token in (pad_token, unk_token):
        if token is not None and token not in specials:
            specials.append(token)

    if max_size is not None:
        if max_size < 0:
            msg = f"max_size must be non-negative, got {max_size}"
            raise ValueError(msg)
        if max_size < len(specials):
            msg = (
                f"max_size ({max_size}) is smaller than the number of "
                f"special tokens ({len(specials)})"
            )
            raise ValueError(msg)

    counts: Counter[str] = Counter()
    for sequence in token_sequences:
        counts.update(sequence)

    # Special tokens are reserved explicitly and never counted as corpus tokens.
    for token in specials:
        counts.pop(token, None)

    # Deterministic order: most frequent first, ties broken by token text.
    candidates = sorted(
        (token for token, count in counts.items() if count >= min_frequency),
        key=lambda token: (-counts[token], token),
    )

    if max_size is not None:
        candidates = candidates[: max_size - len(specials)]

    ordered_tokens = specials + candidates
    token_to_id = {token: index for index, token in enumerate(ordered_tokens)}

    return Vocabulary(
        token_to_id=token_to_id,
        unk_token=unk_token,
        pad_token=pad_token,
    )
