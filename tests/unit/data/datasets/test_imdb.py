"""Tests for polaris.data.datasets.imdb.

All tests run offline against an in-memory fake backend. The real
Hugging Face ``datasets`` library is never invoked.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Final, cast
from unittest.mock import patch

import pytest

from polaris.core.types import DatasetSplit
from polaris.data.datasets.imdb import IMDBDataset
from polaris.data.sample import TextSample
from polaris.errors import DatasetError, InvalidSplitError

type Row = Mapping[str, str | int]

_LOAD_DATASET: Final = "datasets.load_dataset"

_ROWS: Final[tuple[Row, ...]] = (
    {"text": "great movie", "label": 1},
    {"text": "terrible movie", "label": 0},
    {"text": "fine movie", "label": 1},
)


class FakeHFDataset:
    """Minimal in-memory stand-in for a Hugging Face dataset.

    Implements exactly the backend surface IMDBDataset relies on:
    ``__len__``, integer ``__getitem__`` returning a row mapping,
    and ``__iter__`` over row mappings.
    """

    def __init__(self, rows: Sequence[Row]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Row:
        return self._rows[index]

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)


def make_dataset(
    split: DatasetSplit = "train",
    rows: Sequence[Row] = _ROWS,
) -> IMDBDataset:
    """Construct an IMDBDataset backed by the fake backend."""
    with patch(_LOAD_DATASET, return_value=FakeHFDataset(rows)):
        return IMDBDataset(split=split)


@pytest.fixture
def dataset() -> IMDBDataset:
    return make_dataset()


# ============================================================
# Construction
# ============================================================


def test_default_split_is_train(dataset: IMDBDataset) -> None:
    assert dataset.split == "train"


def test_explicit_test_split() -> None:
    assert make_dataset(split="test").split == "test"


def test_backend_receives_expected_arguments() -> None:
    with patch(_LOAD_DATASET, return_value=FakeHFDataset(_ROWS)) as mock_load:
        IMDBDataset(split="test")

    mock_load.assert_called_once_with("stanfordnlp/imdb", split="test")


def test_invalid_split_raises_invalid_split_error() -> None:
    with pytest.raises(InvalidSplitError, match="Invalid split 'banana'"):
        IMDBDataset(split=cast(DatasetSplit, "banana"))


def test_invalid_split_is_raised_before_backend_load() -> None:
    with patch(_LOAD_DATASET) as mock_load, pytest.raises(InvalidSplitError):
        IMDBDataset(split=cast(DatasetSplit, "banana"))

    mock_load.assert_not_called()


def test_backend_failure_is_wrapped_in_dataset_error() -> None:
    error = RuntimeError("boom")

    with (
        patch(_LOAD_DATASET, side_effect=error),
        pytest.raises(DatasetError, match="boom") as exc_info,
    ):
        IMDBDataset()

    assert exc_info.value.__cause__ is error


# ============================================================
# Properties
# ============================================================


def test_name(dataset: IMDBDataset) -> None:
    assert dataset.name == "imdb"


def test_label_names(dataset: IMDBDataset) -> None:
    assert dataset.label_names == ("neg", "pos")


# ============================================================
# Collection protocol
# ============================================================


def test_len(dataset: IMDBDataset) -> None:
    assert len(dataset) == 3


def test_getitem_returns_text_sample(dataset: IMDBDataset) -> None:
    sample = dataset[0]

    assert isinstance(sample, TextSample)
    assert sample.text == "great movie"
    assert sample.label == 1


def test_negative_indexing(dataset: IMDBDataset) -> None:
    assert dataset[-1] == dataset[len(dataset) - 1]
    assert dataset[-3] == dataset[0]


def test_repeated_access_returns_equal_samples(
    dataset: IMDBDataset,
) -> None:
    assert dataset[0] == dataset[0]


@pytest.mark.parametrize("index", [3, -4, 100])
def test_out_of_range_index_raises_index_error(
    dataset: IMDBDataset, index: int
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        dataset[index]


def test_bool_index_is_rejected(dataset: IMDBDataset) -> None:
    with pytest.raises(TypeError, match="got bool"):
        dataset[True]


def test_slice_index_is_rejected(dataset: IMDBDataset) -> None:
    with pytest.raises(TypeError, match="does not support slices"):
        dataset[cast(int, slice(0, 2))]


def test_iteration_matches_indexed_access(dataset: IMDBDataset) -> None:
    assert list(dataset) == [dataset[i] for i in range(len(dataset))]


def test_iteration_yields_text_samples(dataset: IMDBDataset) -> None:
    for sample in dataset:
        assert isinstance(sample, TextSample)


def test_labels_are_valid_label_name_indices(
    dataset: IMDBDataset,
) -> None:
    for sample in dataset:
        assert isinstance(sample.label, int)
        assert 0 <= sample.label < len(dataset.label_names)


# ============================================================
# Representation
# ============================================================


def test_repr(dataset: IMDBDataset) -> None:
    assert repr(dataset) == "IMDBDataset(split='train', num_samples=3)"
