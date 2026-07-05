"""
polaris.core.types
==================

Shared types and foundational data structures used throughout Polaris.

This module defines the common vocabulary understood by every major
framework component. It intentionally contains only generic,
framework-wide types.

Guiding Principles
------------------
- Strong typing
- Immutability where practical
- Framework-wide applicability
- No business logic
- No module-specific definitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type DatasetSplit = Literal["train", "test", "unsupervised"]
# ============================================================
# Type Aliases
# ============================================================

type Identifier = str
"""Unique identifier used throughout the framework."""

type Version = str
"""Semantic version string."""

type Tag = str
"""Human-readable classification tag."""

# ============================================================
# Enumerations
# ============================================================


class ComponentType(StrEnum):
    """
    Supported high-level Polaris component categories.

    Every discoverable framework object belongs to exactly one
    component category.
    """

    MODEL = "model"
    TOKENIZER = "tokenizer"
    DATASET = "dataset"
    METRIC = "metric"
    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    PIPELINE = "pipeline"
    PLUGIN = "plugin"


# ============================================================
# Metadata
# ============================================================


@dataclass(frozen=True, slots=True)
class ComponentMetadata:
    """
    Immutable metadata describing a Polaris component.

    Metadata provides identity and descriptive information only.
    It intentionally does not contain runtime state.
    """

    name: Identifier
    component_type: ComponentType
    description: str

    version: Version = "0.1.0a0"

    author: str | None = None

    license: str | None = None

    homepage: str | None = None

    tags: tuple[Tag, ...] = field(default_factory=tuple)
