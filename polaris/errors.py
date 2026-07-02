"""
polaris.errors
==============

Central exception hierarchy for the Polaris framework.

Every public exception raised by Polaris should inherit from
``PolarisError``.

Design Principles
-----------------
- Stable public API
- Clear exception hierarchy
- Backend-independent errors
- Helpful error messages
- Easy to catch at different levels

Examples
--------
Catch every Polaris exception:

>>> try:
...     ...
... except PolarisError:
...     ...

Catch only dataset-related exceptions:

>>> try:
...     ...
... except DatasetError:
...     ...
"""

from __future__ import annotations


class PolarisError(Exception):
    """
    Base exception for every error raised by Polaris.

    Users who wish to catch all framework-specific exceptions should catch
    this type.
    """


# ============================================================
# Dataset Errors
# ============================================================


class DatasetError(PolarisError):
    """
    Base exception for dataset-related errors.
    """


class InvalidSplitError(DatasetError, ValueError):
    """
    Raised when an invalid dataset split is requested.

    Examples
    --------
    >>> IMDBDataset(split="validation")

    Raises
    ------
    InvalidSplitError
    """


class MissingDependencyError(PolarisError, ImportError):
    """
    Raised when an optional dependency required by Polaris is missing.

    Examples
    --------
    >>> IMDBDataset(...)

    Raises
    ------
    MissingDependencyError
    """
