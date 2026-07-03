# Contributing to Polaris

First and foremost, thank you for your interest in contributing to Polaris. We are excited to build a vibrant community around this project.

Polaris is an open-source NLP Engineering Platform focused on building modern NLP systems using production-inspired software engineering practices. Our goal is to create a platform that is not only powerful but also transparent, maintainable, and educational.

---

## Current Project Status

Polaris is currently in its **early, foundational development phase**.

The core architecture is being established, and many foundational components have not yet been implemented. For a detailed view of our development plan, please see the [**Project Roadmap**](ROADMAP.md).

During this stage, we are not actively seeking large feature contributions. However, we welcome:

- Bug reports and architectural discussions via GitHub Issues.
- Feedback on the project's direction and design.
- Documentation improvements.

---

## Guiding Principles

Every contribution should prioritize:

- **Readability**: Write clear, understandable code and documentation.
- **Maintainability**: Build components that are easy to debug, extend, and refactor.
- **Reproducibility**: Ensure that results and behaviors can be consistently reproduced.
- **Modularity**: Design with clear boundaries and minimal coupling.
- **Type Safety**: Use modern Python type hints to improve code robustness.
- **Documentation**: Document code, architecture, and design decisions.
- **Testing**: Provide comprehensive tests for all new functionality.

The goal is to build software that is understandable and maintainable over the long term.

---

## Reporting Issues

If you discover a bug, have an architectural concern, or find an issue in the documentation, please [**open an issue**](https://github.com/cattolatte/Polaris/issues) on GitHub.

Please provide a clear and concise description, including:

- A summary of the issue.
- The expected behavior.
- The actual behavior.
- Steps to reproduce the issue (if applicable).

---

## Pull Requests

While we are focusing on the core architecture, we will begin accepting pull requests for bug fixes and approved features as the project matures.

When submitting a pull request, please ensure it adheres to the following:

- **Scope**: Keep changes focused and well-scoped to a single issue or feature.
- **Structure**: Follow the existing project structure and architectural patterns.
- **Commits**: Write clear, atomic commit messages.
- **Documentation**: Include or update documentation where appropriate.
- **Tests**: Add or update tests to cover any new functionality or bug fixes.

> **Note**: Large architectural changes **must** be discussed in a GitHub Issue before implementation begins. This ensures alignment with the project's long-term vision.

---

## Coding Standards

Polaris follows modern Python best practices and a strict set of coding standards to ensure quality and consistency.

- **Version**: Python 3.12+
- **Formatting**: We will use [**Black**](https://github.com/psf/black) for code formatting and [**Ruff**](https://github.com/astral-sh/ruff) for linting and import sorting.
- **Type Checking**: We will use [**MyPy**](http://mypy-lang.org/) for static type checking. All new code should be fully type-hinted.
- **Testing**: We will use [**Pytest**](https://pytest.org/) for our test suite.
- **Principles**: Adherence to Clean Architecture and SOLID principles is expected.

Configuration for these tools will be provided in the repository.

---

## Code of Conduct

All contributors are expected to read and adhere to the project's [**Code of Conduct**](CODE_OF_CONDUCT.md). We are committed to fostering a welcoming and respectful community.

---

Thank you for helping make Polaris a better platform for everyone.