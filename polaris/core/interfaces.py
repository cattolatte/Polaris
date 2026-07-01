"""
polaris.core.interfaces
========================

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

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """
    Base protocol implemented by every discoverable Polaris component.

    A Component represents any reusable building block within the framework,
    including (but not limited to):

    - Models
    - Tokenizers
    - Datasets
    - Metrics
    - Trainers
    - Evaluators
    - Plugins

    This protocol deliberately defines only identity-related behavior.

    Domain-specific functionality should be introduced through specialized
    protocols rather than extending this interface unnecessarily.
    """

    @property
    def name(self) -> str:
        """
        Return the unique name of the component.

        Examples
        --------
        bert-base
        wordpiece
        ag_news
        accuracy
        """
        ...

    @property
    def description(self) -> str:
        """
        Return a short human-readable description.

        This value is intended for documentation, command-line interfaces,
        and developer tooling.
        """
        ...

    def metadata(self) -> dict[str, Any]:
        """
        Return metadata describing the component.

        Implementations may expose information such as:

        - author
        - version
        - license
        - tags
        - supported tasks
        - framework-specific metadata

        Returns
        -------
        dict[str, Any]
            Component metadata.
        """
        ...
