# Phase 01 — Foundation

> "A strong framework begins with a strong foundation."

**Status:** ✅ Completed (as re-scoped)

---

## Scope Revision

This document was written before the roadmap adopted its
**"infrastructure when needed"** rule. The original plan listed configuration,
logging, and the CLI as Phase 01 deliverables. Those were subsequently
**deliberately deferred** so that infrastructure is built only when a real
component requires it (see `ROADMAP.md`):

- **Configuration** → introduced with Experiment Tracking (v0.6)
- **Logging** → introduced with the Training Engine (v0.5)
- **CLI** → introduced with Deployment (v0.9)
- **Utilities** → added incrementally, only when a real component needs them

What v0.1.0 actually shipped: the `core` package (protocols + shared types),
the `PolarisError` exception hierarchy, the `Registry`, strict tooling, and CI.
The sections below have been updated to reflect this; the deferred items are
called out rather than removed so the original intent stays visible.

---

# Overview

Phase 01 establishes the engineering foundation of Polaris.

No machine learning models, datasets, or tokenizers will be implemented during this phase.

Instead, this milestone focuses entirely on building the core infrastructure that every future module will depend on.

The quality of this phase directly determines the maintainability, extensibility, and scalability of the framework.

---

# Objectives

The primary objectives of Phase 01 are:

- Establish the core architecture.
- Build reusable framework abstractions.
- Implement a plugin-style registry system.
- Define exception hierarchy.
- Establish testing infrastructure.
- Configure development tooling.

> Configuration management, logging, and the CLI were part of the original
> objectives but were **deferred** to later phases — see the Scope Revision
> note above.

---

# Success Criteria

Phase 01 is considered complete when:

- Core package is implemented.
- Registry system is operational.
- Exception hierarchy is defined.
- Unit tests pass.
- Codebase passes linting and type checking.
- Documentation is updated.

Configuration, CLI, and logging were removed from the completion criteria when
they were deferred to later phases (see the Scope Revision note above).

Only after these requirements are met will Polaris reach **v0.1.0-alpha**.

---

# Architecture

The following modules are actively developed during this phase.

```
polaris/

core/
registry/
utils/

cli.py
version.py
```

All remaining packages remain empty until their corresponding development phases.

---

# Module Responsibilities

## core/

The foundation of Polaris.

Responsibilities:

- Abstract interfaces
- Base classes
- Shared protocols
- Framework exceptions
- Framework constants

The core module must remain independent and should never depend on higher-level modules.

---

## registry/

Provides dynamic registration of framework components.

Initially supports registration of:

- Models
- Tokenizers
- Datasets
- Metrics

Future modules will integrate with this registry rather than relying on hard-coded implementations.

---

## utils/

Contains reusable utility functions shared throughout the framework.

Examples include:

- File helpers
- Random seed utilities
- Version helpers
- Environment detection
- Common validation helpers

Utilities must remain generic.

Domain-specific logic belongs elsewhere.

---

## cli.py

Provides Polaris' command-line interface.

Initial commands may include:

```
polaris --version

polaris info

polaris doctor
```

The CLI will expand as the framework grows.

---

# Dependency Rules

```
core
    ↓
registry
    ↓
utils
    ↓
cli
```

Rules:

- Core depends on nothing.
- Registry depends only on Core.
- Utilities may depend on Core.
- CLI may depend on all lower layers.
- No circular dependencies.

---

# Public APIs

The following APIs will be exposed during Phase 01.

Registry

```python
Registry()

register()
get()
contains()
remove()
clear()
list()
```

Typed helpers such as `register_model()` / `register_dataset()` appeared in the
original sketch but were **not** built. Per the "concrete before abstract" rule
they will be added only when a real caller needs them.

Version

```python
__version__
```

Configuration, Logging, and CLI APIs (`load_config()`, `get_logger()`,
`polaris ...`) are **deferred** to later phases — see the Scope Revision note.

---

# Engineering Decisions

## Python 3.12+

Reason

- Improved performance
- Better typing support
- Pattern matching
- Long-term support

---

## uv

Reason

- Extremely fast package management
- Modern Python workflow
- Excellent lockfile support

---

## Hatchling

Reason

- Lightweight build backend
- Modern packaging
- Excellent library support

---

## Black

Reason

Automatic code formatting.

No discussions.

No formatting debates.

---

## Ruff

Reason

Fast linting.

Replaces multiple legacy tools.

---

## MyPy

Reason

Static type checking.

Improves maintainability.

---

## Pytest

Reason

Industry-standard Python testing framework.

---

## Pydantic v2

Reason

Configuration validation.

Type safety.

Serialization.

Hydra will be evaluated during later experiment-tracking phases.

---

## Typer

Reason

Modern CLI framework.

Strong type hints.

Simple API.

---

## Python logging

Reason

Standard library.

Widely adopted.

Avoids unnecessary dependencies.

---

# File Structure

```
polaris/

core/

registry/

utils/

cli.py

version.py

tests/

unit/

core/

registry/

utils/
```

No additional modules are introduced during this phase.

---

# Testing Strategy

Every public component must include unit tests.

Testing goals include:

- Registry registration
- Registry lookup
- Configuration validation
- CLI execution
- Utility functions
- Exception handling

Target:

High confidence in framework infrastructure before expanding functionality.

---

# Coding Standards

Every contribution should follow:

- Black formatting
- Ruff linting
- MyPy type checking
- Comprehensive docstrings
- Type hints
- Small focused functions
- Clear naming

Readability is preferred over clever implementations.

---

# Deliverables

- [x] Core infrastructure
- [x] Registry system
- [x] Exception hierarchy
- [x] Tests
- [x] Documentation
- [x] CI configuration
- [ ] Configuration system — deferred to v0.6
- [ ] Logging — deferred to v0.5
- [ ] CLI — deferred to v0.9
- [ ] Utilities — added incrementally when a real component needs them

---

# Out of Scope

The following are intentionally excluded from Phase 01.

- Datasets
- Tokenizers
- Models
- Training
- Evaluation
- Deployment
- Visualization
- Experiment tracking

These components will be developed in future milestones.

---

# Expected Outcome

At the end of Phase 01, Polaris will possess a robust engineering foundation capable of supporting all future development.

No end-user machine learning functionality will exist yet.

However, the framework will provide the architectural backbone upon which every subsequent component is built.

---

> "Strong software is built on deliberate engineering decisions, not accidental architecture."