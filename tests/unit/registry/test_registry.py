"""
Unit tests for the Polaris registry.
"""

from __future__ import annotations

import pytest

from polaris.core.types import ComponentMetadata, ComponentType
from polaris.registry import Registry


class DummyComponent:
    """Simple component used for registry testing."""

    def __init__(
        self,
        name: str,
        component_type: ComponentType = ComponentType.PLUGIN,
    ) -> None:
        self._metadata = ComponentMetadata(
            name=name,
            component_type=component_type,
            description=f"{name} component",
            version="0.1.0a0",
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._metadata


def test_register_component() -> None:
    """Components should be registered successfully."""

    registry = Registry()
    component = DummyComponent("dummy")

    registry.register(component)

    assert len(registry) == 1


def test_get_registered_component() -> None:
    """Registered components should be retrievable."""

    registry = Registry()
    component = DummyComponent("dummy")

    registry.register(component)

    assert registry.get("dummy") is component


def test_contains_registered_component() -> None:
    """Membership checks should succeed."""

    registry = Registry()
    registry.register(DummyComponent("dummy"))

    assert registry.contains("dummy")
    assert "dummy" in registry


def test_duplicate_registration_raises_error() -> None:
    """Duplicate component names are not allowed."""

    registry = Registry()

    registry.register(DummyComponent("dummy"))

    with pytest.raises(ValueError):
        registry.register(DummyComponent("dummy"))


def test_remove_component() -> None:
    """Registered components should be removable."""

    registry = Registry()

    registry.register(DummyComponent("dummy"))

    registry.remove("dummy")

    assert len(registry) == 0


def test_clear_registry() -> None:
    """The registry should be completely cleared."""

    registry = Registry()

    registry.register(DummyComponent("a"))
    registry.register(DummyComponent("b"))

    registry.clear()

    assert len(registry) == 0


def test_list_all_components() -> None:
    """Listing without filters should return all metadata."""

    registry = Registry()

    registry.register(DummyComponent("a"))
    registry.register(DummyComponent("b"))

    metadata = registry.list()

    assert len(metadata) == 2


def test_filter_by_component_type() -> None:
    """Listing should support component type filtering."""

    registry = Registry()

    registry.register(
        DummyComponent(
            "bert",
            ComponentType.MODEL,
        )
    )

    registry.register(
        DummyComponent(
            "bpe",
            ComponentType.TOKENIZER,
        )
    )

    models = registry.list(ComponentType.MODEL)

    assert len(models) == 1
    assert models[0].name == "bert"


def test_registry_iteration() -> None:
    """Registry should be iterable."""

    registry = Registry()

    registry.register(DummyComponent("a"))
    registry.register(DummyComponent("b"))

    names = {component.metadata.name for component in registry}

    assert names == {"a", "b"}


def test_missing_component_raises_key_error() -> None:
    """Unknown components should raise KeyError."""

    registry = Registry()

    with pytest.raises(KeyError):
        registry.get("missing")
