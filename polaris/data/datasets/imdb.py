"""
polaris.data.datasets.imdb
==========================

IMDB movie review dataset.

Backed by the Hugging Face ``datasets`` library, but exposes only
Polaris-native abstractions. Users never interact with the backend.

Design Principles
-----------------
- Read-only collection
- Eager dataset loading
- Lazy TextSample conversion
- Backend abstraction
- Strong typing
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from polaris.core.types import DatasetSplit
from polaris.data.base import Dataset
from polaris.data.sample import TextSample
from polaris.errors import (
    DatasetError,
    InvalidSplitError,
    MissingDependencyError,
)

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

_VALID_SPLITS: Final[tuple[DatasetSplit, ...]] = ("train", "test", "unsupervised")
_LABEL_NAMES: Final[tuple[str, str]] = ("neg", "pos")


class IMDBDataset(Dataset):
    """
    IMDB movie review dataset.

    IMDBDataset satisfies the Polaris ``Dataset`` protocol.

    Parameters
    ----------
    split:
        Dataset split to load. One of ``"train"``, ``"test"``, or
        ``"unsupervised"`` (50,000 unlabeled reviews, label ``-1``, for
        self-supervised pretraining).

    Notes
    -----
    The backend dataset is loaded eagerly during construction.
    ``TextSample`` conversion happens lazily on access, so repeated
    access returns equal (but not identical) samples.

    Examples
    --------
    >>> dataset = IMDBDataset(split="train")
    >>> sample = dataset[0]
    """

    def __init__(self, split: DatasetSplit = "train") -> None:
        if split not in _VALID_SPLITS:
            raise InvalidSplitError(
                f"Invalid split {split!r}. Expected one of {_VALID_SPLITS}."
            )

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'datasets' package is required to use IMDBDataset.\n\n"
                "Install it using:\n\n"
                "    uv sync --extra datasets"
            ) from exc

        self._split: DatasetSplit = split

        try:
            # Canonical Hugging Face repo id. Newer datasets/huggingface_hub
            # require the "namespace/name" form and reject the bare "imdb".
            self._data: HFDataset = load_dataset(
                "stanfordnlp/imdb",
                split=split,
            )
        except Exception as exc:
            raise DatasetError(
                f"Failed to load the IMDB {split!r} split: {exc}"
            ) from exc

    # ============================================================
    # Properties
    # ============================================================

    @property
    def name(self) -> str:
        """Stable dataset identifier."""
        return "imdb"

    @property
    def split(self) -> DatasetSplit:
        """Loaded dataset split."""
        return self._split

    @property
    def label_names(self) -> tuple[str, str]:
        """Human-readable class labels, indexed by label id."""
        return _LABEL_NAMES

    # ============================================================
    # Conversion
    # ============================================================

    def _to_text_sample(self, row: Mapping[str, Any]) -> TextSample:
        """Single conversion point from a backend row to a TextSample."""
        return TextSample(
            text=row["text"],
            label=row["label"],
        )

    # ============================================================
    # Collection Protocol
    # ============================================================

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> TextSample:
        if isinstance(index, slice):
            raise TypeError("IMDBDataset does not support slices; index with an int.")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"Dataset indices must be integers, " f"got {type(index).__name__}."
            )

        length = len(self)
        normalized = index + length if index < 0 else index

        if not 0 <= normalized < length:
            raise IndexError(
                f"Index {index} is out of range for dataset of "
                f"length {length} (valid range: 0..{length - 1})."
            )

        return self._to_text_sample(self._data[normalized])

    def __iter__(self) -> Iterator[TextSample]:
        for row in self._data:
            yield self._to_text_sample(row)

    # ============================================================
    # Representation
    # ============================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self._split!r}, "
            f"num_samples={len(self)})"
        )
