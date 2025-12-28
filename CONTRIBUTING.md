# Contributing to ReasonKit Core

Thank you for your interest in contributing to ReasonKit Core! Every contribution helps make AI reasoning more structured, auditable, and reliable.

Your time is valuable, so we've kept this guide focused and actionable.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Guidelines](#code-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Getting Help](#getting-help)
- [Recognition](#recognition)

---

## Code of Conduct

We are committed to providing a welcoming environment for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

**TL;DR:** Be respectful, be constructive, be inclusive.

---

## Ways to Contribute

| Type                 | Impact | Good For                 |
| -------------------- | ------ | ------------------------ |
| **Report bugs**      | High   | Everyone                 |
| **Fix bugs**         | High   | Developers               |
| **Improve docs**     | Medium | Writers, newcomers       |
| **Add features**     | High   | Experienced contributors |
| **Answer questions** | Medium | Community members        |
| **Review PRs**       | High   | Experienced developers   |

### Good First Issues

New to ReasonKit? Look for these labels:

- `good first issue` - Beginner-friendly, well-scoped
- `documentation` - Improve docs, fix typos
- `help wanted` - Community help needed

Browse: [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue")

---

## Development Setup

### Prerequisites

- Rust 1.74+ ([rustup.rs](https://rustup.rs/))
- Git 2.30+

### Quick Start

```bash
# Clone the repository
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core

# Build
cargo build

# Run tests
cargo test

# Verify everything works
cargo clippy -- -D warnings
cargo fmt --check
```

If all commands pass, you're ready to contribute.

### Optional Setup

```bash
# Install helpful tools
cargo install cargo-tarpaulin  # Coverage
cargo install cargo-audit      # Security audit
```

---

## Code Guidelines

### Rust Style

We use `rustfmt` and `clippy` for consistent code style.

```bash
# Format your code
cargo fmt

# Check for lint issues (must pass with zero warnings)
cargo clippy -- -D warnings
```

### Key Rules

| Rule                                  | Rationale                                                   |
| ------------------------------------- | ----------------------------------------------------------- |
| **No `unsafe` without justification** | Document the rationale in comments if you must use `unsafe` |
| **Tests required for new features**   | All new functionality needs test coverage                   |
| **Document public APIs**              | Use `///` doc comments for public items                     |
| **Use `Result<T, E>` for errors**     | Prefer `anyhow` for apps, `thiserror` for libraries         |

### Example Documentation

```rust
/// Performs hybrid search combining BM25 and vector similarity.
///
/// # Arguments
///
/// * `query` - The search query string
/// * `top_k` - Maximum number of results to return
///
/// # Returns
///
/// A vector of search results sorted by relevance.
///
/// # Errors
///
/// Returns an error if the query is empty or index is unavailable.
pub fn hybrid_search(query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
    // Implementation
}
```

### Quality Gates

All code must pass these gates before merge:

| Gate   | Command                       | Requirement     |
| ------ | ----------------------------- | --------------- |
| Build  | `cargo build --release`       | Exit 0          |
| Lint   | `cargo clippy -- -D warnings` | 0 warnings      |
| Format | `cargo fmt --check`           | Pass            |
| Tests  | `cargo test --all-features`   | 100% pass       |
| Bench  | `cargo bench`                 | < 5% regression |

---

## Pull Request Process

### Workflow

1. **Fork** the repository
2. **Branch** from `main` (`git checkout -b feature/your-feature`)
3. **Make changes** following the code guidelines
4. **Test** locally (`cargo test`)
5. **Commit** with conventional commits
6. **Push** and open a PR

### Branch Naming

```
feature/   - New features (feature/hybrid-search)
fix/       - Bug fixes (fix/chunk-splitting)
docs/      - Documentation (docs/api-reference)
perf/      - Performance (perf/vector-optimization)
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat(search): add BM25 scoring algorithm
fix(ingestion): handle empty PDF pages
docs(readme): update installation instructions
perf(embedding): batch vector operations
test(retrieval): add edge case coverage
```

### PR Checklist

Before submitting:

- [ ] All quality gates pass locally
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

### Review Process

1. **CI checks** run automatically
2. **Maintainer reviews** within 1 week
3. **Address feedback** if requested
4. **Merge** after approval

---

## Issue Guidelines

### Before Opening an Issue

1. **Search existing issues** - It may already be reported
2. **Update to latest version** - The issue may be fixed
3. **Prepare minimal reproduction** - Helps us fix it faster

### Bug Reports

Include:

- ReasonKit version
- Rust version (`rustc --version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

Include:

- Use case / problem statement
- Proposed solution
- Alternatives considered

### Use Issue Templates

We provide templates for bugs and features. Please use them.

---

## Getting Help

| Channel                | Best For                         | Link                                                                   |
| ---------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| **Discord**            | Quick questions, community chat  | [Join Discord](https://discord.gg/reasonkit)                           |
| **GitHub Discussions** | Ideas, Q&A, longer conversations | [Discussions](https://github.com/reasonkit/reasonkit-core/discussions) |
| **GitHub Issues**      | Bugs, feature requests           | [Issues](https://github.com/reasonkit/reasonkit-core/issues)           |

### Response Times

- Issues: Within 48 hours
- PRs: Within 1 week
- Security: Within 24 hours (email security@reasonkit.sh)

---

## Recognition

Contributors are recognized in:

- **Release notes** for significant contributions
- **GitHub contributors page**
- **CONTRIBUTORS.md** (coming soon)

We value every contribution, from typo fixes to major features.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

---

## Thank You

Every contribution makes ReasonKit better. Whether you're fixing a typo or implementing a major feature, your work matters.

Questions? Open a [Discussion](https://github.com/reasonkit/reasonkit-core/discussions) or join our [Discord](https://discord.gg/reasonkit).

---

_"Designed, Not Dreamed. Turn Prompts into Protocols."_

https://reasonkit.sh
