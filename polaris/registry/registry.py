"""
polaris.registry.registry
=========================

Core registry implementation for Polaris.

The registry provides centralized discovery and management of
framework components.

Design Principles
-----------------
- Strong typing
- Explicit registration
- No global mutable state
- Fast lookup
- Predictable behavior
"""

from __future__ import annotations

from collections.abc import Iterator

from polaris.core.interfaces import Component
from polaris.core.types import ComponentMetadata, ComponentType, Identifier


class Registry:
    """
    Registry of discoverable Polaris components.

    Components are uniquely identified by their metadata.name.
    """

    def __init__(self) -> None:
        self._components: dict[Identifier, Component] = {}

    def register(self, component: Component) -> None:
        """
        Register a component.

        Raises
        ------
        ValueError
            If another component with the same name already exists.
        """

        identifier = component.metadata.name

        if identifier in self._components:
            raise ValueError(f"Component '{identifier}' is already registered.")

        self._components[identifier] = component

    def get(self, identifier: Identifier) -> Component:
        """
        Retrieve a registered component.

        Raises
        ------
        KeyError
            If the component is not registered.
        """

        return self._components[identifier]

    def contains(self, identifier: Identifier) -> bool:
        """Return True if a component is registered."""

        return identifier in self._components

    def remove(self, identifier: Identifier) -> None:
        """Remove a registered component."""

        del self._components[identifier]

    def clear(self) -> None:
        """Remove every registered component."""

        self._components.clear()

    def list(
        self,
        component_type: ComponentType | None = None,
    ) -> list[ComponentMetadata]:
        """
        Return metadata for registered components.

        When a component type is provided, only matching components
        are returned.
        """

        metadata = [component.metadata for component in self._components.values()]

        if component_type is None:
            return metadata

        return [item for item in metadata if item.component_type is component_type]

    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str):
            return False

        return identifier in self._components

    def __iter__(self) -> Iterator[Component]:
        return iter(self._components.values())
