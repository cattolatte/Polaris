"""
Unit tests for Polaris core interfaces.
"""

from __future__ import annotations

from polaris.core.interfaces import Component


class DummyComponent:
    """Simple implementation used for protocol testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy component"

    def metadata(self) -> dict[str, str]:
        return {
            "version": "1.0.0",
        }


def test_component_protocol_runtime_check() -> None:
    """Objects implementing the protocol should satisfy isinstance()."""

    component = DummyComponent()

    assert isinstance(component, Component)


def test_component_name() -> None:
    """Component should expose a name."""

    component = DummyComponent()

    assert component.name == "dummy"


def test_component_description() -> None:
    """Component should expose a description."""

    component = DummyComponent()

    assert component.description == "Dummy component"


def test_component_metadata() -> None:
    """Component metadata should be accessible."""

    component = DummyComponent()

    assert component.metadata()["version"] == "1.0.0"
