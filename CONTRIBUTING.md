# Contributing to ReasonKit Core

We welcome contributions to ReasonKit! This guide will help you get started.

## Development Setup

1.  **Prerequisites:**
    *   Rust 1.75+
    *   `cargo`
    *   `uv` (for Python bindings)

2.  **Clone & Build:**
    ```bash
    git clone https://github.com/ReasonKit/reasonkit-core.git
    cd reasonkit-core
    cargo build
    ```

## Workflow

1.  **Fork** the repository.
2.  **Create a branch** for your feature/fix (`git checkout -b feature/my-feature`).
3.  **Code** your changes.
    *   Follow Rust idioms.
    *   Add tests for new functionality.
4.  **Test** locally:
    ```bash
    cargo test
    cargo clippy -- -D warnings
    cargo fmt --check
    ```
5.  **Submit a Pull Request**.

## Style Guide

*   Use `cargo fmt`.
*   Document public APIs with Rustdoc (`///`).
*   Keep functions small and focused.
*   Use `Result` for error handling (no `unwrap` in production code).

## Code of Conduct

Please adhere to the [Contributor Covenant](https://www.contributor-covenant.org/).
