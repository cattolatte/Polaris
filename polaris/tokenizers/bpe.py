"""Byte Pair Encoding (BPE) tokenization, implemented from scratch.

BPE is a subword tokenizer: it learns to merge frequently-adjacent symbols, so
rare or unseen words are split into known subwords instead of collapsing to
``<unk>``. This is the classic word-level algorithm (Sennrich et al.), written by
hand on plain data structures — the point is to *understand* subwording.

Training operates on unique words weighted by their frequency (not every
occurrence), which is the standard efficiency of word-level BPE. It is still a
readable reference, not an optimized implementation.

Design Principles
-----------------
- ``train_bpe`` (construction) is kept separate from ``BPETokenizer`` (runtime),
  mirroring how ``build_vocabulary`` is separate from ``Vocabulary``.
- ``BPETokenizer`` is the second concrete :class:`~polaris.tokenizers.tokenizer.
  Tokenizer`, validating that protocol; it introduces no new abstraction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from polaris.tokenizers.encoding import Encoding
from polaris.tokenizers.vocabulary import Vocabulary

__all__ = ["BPETokenizer", "train_bpe"]


def _merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Merge every left-to-right occurrence of ``pair`` in ``symbols``."""
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index < len(symbols) - 1
            and symbols[index] == pair[0]
            and symbols[index + 1] == pair[1]
        ):
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def train_bpe(
    token_sequences: Iterable[Iterable[str]],
    *,
    vocab_size: int,
    unk_token: str | None = None,
    pad_token: str | None = None,
    mask_token: str | None = None,
    cls_token: str | None = None,
    sep_token: str | None = None,
    end_of_word: str = "</w>",
    min_frequency: int = 1,
) -> BPETokenizer:
    """Learn a BPE tokenizer from a corpus of words.

    Each word is represented as its characters plus an end-of-word marker; the
    most frequent adjacent symbol pair (weighted by word frequency) is merged
    repeatedly, recording each merge in order, until the vocabulary reaches
    ``vocab_size``.

    Parameters
    ----------
    token_sequences : Iterable[Iterable[str]]
        The corpus as sequences of words (e.g. whitespace-split documents).
    vocab_size : int
        Target vocabulary size (special tokens and base characters count toward
        it; it bounds how many merges are learned).
    unk_token, pad_token, mask_token, cls_token, sep_token : str, optional
        Special tokens, reserved first and contiguously (in that order) and passed
        to the resulting vocabulary, so adding one never renumbers other symbols.
        ``mask_token`` is for MLM; ``cls_token`` / ``sep_token`` for pair models.
    end_of_word : str, default "</w>"
        Marker appended to each word so word boundaries survive tokenization.
    min_frequency : int, default 1
        Merges are learned only from words appearing at least this many times.
        Rarer words are skipped for merge-learning (keeping training tractable on
        large corpora), but their characters still count toward coverage.

    Returns
    -------
    BPETokenizer
        The trained tokenizer.
    """
    if min_frequency < 1:
        raise ValueError(f"min_frequency must be at least 1, got {min_frequency}")

    word_frequencies: Counter[str] = Counter()
    for sequence in token_sequences:
        word_frequencies.update(sequence)

    specials: list[str] = []
    for token in (pad_token, unk_token, mask_token, cls_token, sep_token):
        if token is not None and token not in specials:
            specials.append(token)

    # Character coverage comes from every word; merges are learned only from
    # words meeting `min_frequency`.
    base_vocab: set[str] = {end_of_word}
    for word in word_frequencies:
        base_vocab.update(word)

    word_symbols = {
        word: list(word) + [end_of_word]
        for word, frequency in word_frequencies.items()
        if frequency >= min_frequency
    }

    current_vocab = set(specials) | base_vocab
    merges: list[tuple[str, str]] = []
    while len(current_vocab) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, symbols in word_symbols.items():
            frequency = word_frequencies[word]
            for left, right in zip(symbols, symbols[1:], strict=False):
                pair_counts[(left, right)] += frequency
        if not pair_counts:
            break
        # Most frequent pair; ties broken deterministically by the pair itself.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        current_vocab.add(best_pair[0] + best_pair[1])
        for word in word_symbols:
            word_symbols[word] = _merge_pair(word_symbols[word], best_pair)

    # Assign ids: special tokens first, then base symbols, then merged symbols.
    ordered: list[str] = list(specials)
    for symbol in sorted(base_vocab):
        if symbol not in ordered:
            ordered.append(symbol)
    for left, right in merges:
        symbol = left + right
        if symbol not in ordered:
            ordered.append(symbol)

    vocabulary = Vocabulary(
        token_to_id={token: index for index, token in enumerate(ordered)},
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
        cls_token=cls_token,
        sep_token=sep_token,
    )
    return BPETokenizer(vocabulary, merges, end_of_word=end_of_word)


class BPETokenizer:
    """A subword tokenizer that applies learned BPE merges.

    Satisfies the :class:`~polaris.tokenizers.tokenizer.Tokenizer` protocol.

    Parameters
    ----------
    vocabulary : Vocabulary
        The subword vocabulary (typically produced by :func:`train_bpe`).
    merges : Sequence[tuple[str, str]]
        The learned merges, in application order.
    end_of_word : str, default "</w>"
        The end-of-word marker used during training.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        merges: Sequence[tuple[str, str]],
        *,
        end_of_word: str = "</w>",
    ) -> None:
        self._vocabulary = vocabulary
        self._merges = tuple(merges)
        self._ranks = {pair: rank for rank, pair in enumerate(self._merges)}
        self._end_of_word = end_of_word

    @property
    def vocabulary(self) -> Vocabulary:
        """Vocabulary: the subword vocabulary."""
        return self._vocabulary

    def _bpe(self, word: str) -> tuple[str, ...]:
        """Apply the learned merges to a single word."""
        symbols = list(word) + [self._end_of_word]
        while len(symbols) > 1:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for left, right in zip(symbols, symbols[1:], strict=False):
                rank = self._ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (left, right)
            if best_pair is None:
                break
            symbols = _merge_pair(symbols, best_pair)
        return tuple(symbols)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split ``text`` into subword tokens."""
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self._bpe(word))
        return tuple(tokens)

    def encode(self, text: str) -> Encoding:
        """Tokenize ``text`` and map subwords to ids (with unknown fallback)."""
        tokens = self.tokenize(text)
        ids = tuple(self._vocabulary.get_id(token) for token in tokens)
        return Encoding(ids=ids, tokens=tokens)

    def decode(self, ids: Sequence[int]) -> str:
        """Reconstruct text from ids by joining subwords and restoring spaces."""
        tokens = [self._vocabulary.lookup_token(token_id) for token_id in ids]
        return "".join(tokens).replace(self._end_of_word, " ").strip()
