"""
Unit tests for Polaris core interfaces.
"""

from __future__ import annotations

from polaris.core.interfaces import Component
from polaris.core.types import ComponentMetadata, ComponentType


class DummyComponent:
    """Simple implementation used for protocol testing."""

    @property
    def metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="dummy",
            component_type=ComponentType.PLUGIN,
            description="Dummy component",
            version="0.1.0a0",
            author="Polaris",
            tags=("test",),
        )


def test_component_protocol_runtime_check() -> None:
    """Objects implementing the protocol should satisfy isinstance()."""

    component = DummyComponent()

    assert isinstance(component, Component)


def test_component_metadata_type() -> None:
    """Metadata should be strongly typed."""

    component = DummyComponent()

    assert isinstance(component.metadata, ComponentMetadata)


def test_component_metadata_name() -> None:
    """Metadata should expose the component name."""

    component = DummyComponent()

    assert component.metadata.name == "dummy"


def test_component_metadata_description() -> None:
    """Metadata should expose the component description."""

    component = DummyComponent()

    assert component.metadata.description == "Dummy component"


def test_component_metadata_contents() -> None:
    """Metadata should expose all expected values."""

    metadata = DummyComponent().metadata

    assert metadata.name == "dummy"
    assert metadata.component_type is ComponentType.PLUGIN
    assert metadata.version == "0.1.0a0"
    assert metadata.author == "Polaris"
    assert metadata.tags == ("test",)
