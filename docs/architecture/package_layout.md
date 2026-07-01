# Polaris Package Layout

polaris/

├── core/
├── registry/
├── data/
├── tokenizers/
├── models/
├── engine/
├── experiments/
├── evaluation/
├── deployment/
├── visualization/
├── plugins/
└── utils/

Each package has a single responsibility.

The architecture follows:

Core
↓
Registry
↓
Domain Modules
↓
CLI

No module should depend on implementation details from another module unless explicitly designed to do so.
