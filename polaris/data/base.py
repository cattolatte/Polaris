"""
polaris.data.base
=================

Core dataset abstractions for Polaris.

This module defines the protocol implemented by every dataset supported by
Polaris.

A dataset represents an immutable collection of text samples.

The protocol intentionally remains minimal and focuses only on behaviour
required by the rest of the framework.

Design Principles
-----------------
- Protocol-oriented design
- Read-only collections
- Strong typing
- Model agnostic
- Tokenizer agnostic
- Minimal abstraction

Concrete dataset implementations live under
``polaris.data.datasets``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from polaris.data.sample import TextSample


@runtime_checkable
class Dataset(Protocol):
    """
    Protocol implemented by every Polaris dataset.

    A dataset behaves as a read-only collection of ``TextSample`` objects.

    Examples
    --------
    >>> len(dataset)

    >>> dataset[0]

    >>> for sample in dataset:
    ...     ...
    """

    @property
    def name(self) -> str:
        """
        Human-readable dataset name.

        Examples
        --------
        IMDB

        AG News

        Yelp Reviews
        """
        ...

    def __len__(self) -> int:
        """
        Return the total number of samples contained in the dataset.
        """
        ...

    def __getitem__(self, index: int) -> TextSample:
        """
        Retrieve a single sample.

        Parameters
        ----------
        index:
            Zero-based sample index.

        Returns
        -------
        TextSample
            Requested sample.

        Raises
        ------
        IndexError
            If the index is outside the dataset bounds.
        """
        ...

    def __iter__(self) -> Iterator[TextSample]:
        """
        Iterate over every sample contained in the dataset.
        """
        ...
