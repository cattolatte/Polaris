"""
polaris.data.sample
===================

Core data structures used throughout the Polaris data subsystem.

This module defines the fundamental representation of a single text sample.

Every NLP workflow inside Polaris begins with a TextSample.

The object intentionally contains only raw textual information and associated
labels. It has no knowledge of tokenization, numerical encoding, neural
networks, or training.

Design Principles
-----------------
- Immutable
- Strongly typed
- Framework agnostic
- Tokenizer agnostic
- Model agnostic
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

type Label = int | str


@dataclass(frozen=True, slots=True)
class TextSample:
    """
    Represents a single supervised NLP example.

    A ``TextSample`` is the fundamental unit of data exchanged throughout the
    Polaris framework. Every dataset implementation returns ``TextSample``
    objects, allowing downstream components to remain independent from any
    specific dataset backend.

    Parameters
    ----------
    text:
        Raw textual input.

    label:
        Ground-truth label associated with the text.

        The label representation is intentionally generic to support multiple
        NLP tasks beyond text classification.

    metadata:
        Optional immutable metadata describing the sample.

    Examples
    --------
    >>> TextSample(
    ...     text="This movie was fantastic!",
    ...     label="positive",
    ... )

    >>> TextSample(
    ...     text="Apple announces new MacBook.",
    ...     label=3,
    ... )
    """

    text: str

    label: Label

    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """
        Ensure metadata is immutable.

        Even though the dataclass itself is frozen, mutable dictionaries would
        otherwise allow indirect mutation. Metadata is therefore converted into
        a read-only mapping.
        """
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
