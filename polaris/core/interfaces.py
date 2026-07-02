"""
polaris.core.interfaces
=======================

Core framework interfaces for Polaris.

This module defines the foundational behavioral contracts that every
major Polaris component is expected to follow.

These interfaces intentionally avoid implementation details and instead
focus on defining consistent expectations across the framework.

Design Principles
-----------------
- Protocol-oriented design
- Composition over inheritance
- Strong typing
- Minimal abstractions
- No framework-specific business logic

Concrete implementations live in their respective modules.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from polaris.core.types import ComponentMetadata


@runtime_checkable
class Component(Protocol):
    """
    Base protocol implemented by every discoverable Polaris component.

    Every Polaris component exposes immutable metadata describing
    its identity.

    Concrete functionality belongs to specialized protocols such as
    Model, Dataset, Tokenizer, Trainer, Metric, and Pipeline.
    """

    @property
    def metadata(self) -> ComponentMetadata:
        """
        Immutable metadata describing this component.
        """
        ...
