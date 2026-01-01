# Contributing to ReasonKit Core

Thank you for your interest in contributing to ReasonKit Core! Every contribution helps make AI reasoning more structured, auditable, and reliable.

We welcome contributions from everyone, regardless of experience level. This guide will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Getting Help](#getting-help)
- [Recognition](#recognition)

---

## Code of Conduct

We are committed to providing a welcoming environment for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

**Summary:** Be respectful, be constructive, be inclusive. Harassment and exclusionary behavior are not tolerated.

---

## Ways to Contribute

| Type               | Impact | Good For                 | Difficulty   |
| ------------------ | ------ | ------------------------ | ------------ |
| **Report bugs**    | High   | Everyone                 | Beginner     |
| **Fix typos/docs** | Medium | Writers, newcomers       | Beginner     |
| **Fix bugs**       | High   | Developers               | Intermediate |
| **Add tests**      | High   | Detail-oriented devs     | Intermediate |
| **Improve docs**   | Medium | Technical writers        | Intermediate |
| **Add features**   | High   | Experienced contributors | Advanced     |
| **Review PRs**     | High   | Experienced developers   | Advanced     |

### Good First Issues

New to ReasonKit? Look for these labels:

- `good first issue` - Beginner-friendly, well-scoped tasks
- `documentation` - Improve docs, fix typos, add examples
- `help wanted` - Community help needed

Browse: [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue")

---

## Development Setup

### Prerequisites

| Tool     | Version | Installation                        |
| -------- | ------- | ----------------------------------- |
| **Rust** | 1.74+   | [rustup.rs](https://rustup.rs/)     |
| **Git**  | 2.30+   | [git-scm.com](https://git-scm.com/) |

### Quick Start (5 minutes)

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/reasonkit-core.git
cd reasonkit-core

# 2. Add upstream remote
git remote add upstream https://github.com/reasonkit/reasonkit-core.git

# 3. Build the project
cargo build

# 4. Run tests
cargo test

# 5. Verify linting passes
cargo clippy -- -D warnings
cargo fmt --check
```

If all commands pass, you are ready to contribute!

### Recommended Development Tools

```bash
# Install helpful Rust tools (optional but recommended)
cargo install cargo-watch     # Auto-rebuild on file changes
cargo install cargo-tarpaulin # Code coverage
cargo install cargo-audit     # Security vulnerability scanner
cargo install cargo-expand    # Macro expansion viewer
cargo install cargo-nextest   # Faster test runner
```

### IDE Setup

**VS Code (Recommended)**

Install these extensions:

- [Rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) - Rust language support
- [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) - Cargo.toml support
- [Error Lens](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens) - Inline error display

Recommended `settings.json`:

```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.cargo.features": "all",
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

**JetBrains RustRover / CLion**

- Install the Rust plugin
- Enable "Run rustfmt on save"
- Configure Clippy as the default checker

---

## Code Style Guidelines

### The Rust Supremacy Doctrine

ReasonKit Core follows **The Rust Supremacy** - a non-negotiable architectural principle:

| Principle          | Requirement                                    |
| ------------------ | ---------------------------------------------- |
| **Performance**    | All core loops must execute in < 5ms           |
| **Memory Safety**  | No `unsafe` without documented justification   |
| **Type Safety**    | Prefer strong typing over runtime checks       |
| **Error Handling** | Use `Result<T, E>` - no panics in library code |
| **Concurrency**    | Use Rust's ownership model, avoid shared state |

### Formatting and Linting

We use `rustfmt` and `clippy` for consistent code style. These are enforced in CI.

```bash
# Format your code (run before every commit)
cargo fmt

# Check for lint issues (must pass with zero warnings)
cargo clippy -- -D warnings

# For stricter checking (mirrors CI)
cargo clippy --all-targets --all-features -- -D warnings
```

### Naming Conventions

| Item              | Convention       | Example                     |
| ----------------- | ---------------- | --------------------------- |
| Crates            | `snake_case`     | `reasonkit_core`            |
| Modules           | `snake_case`     | `think_tools`               |
| Types/Traits      | `PascalCase`     | `ThinkClient`, `Searchable` |
| Functions/Methods | `snake_case`     | `hybrid_search`             |
| Constants         | `SCREAMING_CASE` | `MAX_RETRIES`               |
| Local Variables   | `snake_case`     | `query_result`              |
| Type Parameters   | Single letter    | `T`, `E`, `R`               |
| Lifetimes         | Short lowercase  | `'a`, `'ctx`                |

### Documentation Standards

All public APIs must be documented with `///` doc comments. For general documentation contributions, please refer to our [Documentation Maintenance Guide](docs/MAINTENANCE.md).

````rust
/// Performs hybrid search combining BM25 and vector similarity.
///
/// This method executes both sparse (BM25) and dense (vector) retrieval,
/// then fuses results using Reciprocal Rank Fusion (RRF).
///
/// # Arguments
///
/// * `query` - The search query string (must not be empty)
/// * `top_k` - Maximum number of results to return (1-1000)
///
/// # Returns
///
/// A vector of `SearchResult` sorted by relevance score (descending).
///
/// # Errors
///
/// Returns `SearchError::EmptyQuery` if the query string is empty.
/// Returns `SearchError::IndexUnavailable` if the search index is not ready.
///
/// # Examples
///
/// ```rust
/// use reasonkit::search::SearchClient;
///
/// let client = SearchClient::new()?;
/// let results = client.hybrid_search("microservices architecture", 10)?;
///
/// for result in results {
///     println!("{}: {:.2}", result.title, result.score);
/// }
/// ```
///
/// # Performance
///
/// Typical latency: < 50ms for 100K documents.
pub fn hybrid_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, SearchError> {
    // Implementation
}
````

### Error Handling Best Practices

```rust
// GOOD: Use thiserror for library errors
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("query cannot be empty")]
    EmptyQuery,

    #[error("search index unavailable: {0}")]
    IndexUnavailable(String),

    #[error("query timeout after {0}ms")]
    Timeout(u64),
}

// GOOD: Use anyhow for application code
fn main() -> anyhow::Result<()> {
    let client = SearchClient::new()?;
    let results = client.search("query")?;
    Ok(())
}

// BAD: Don't panic in library code
pub fn search(query: &str) -> Vec<Result> {
    if query.is_empty() {
        panic!("Query cannot be empty");  // Never do this!
    }
}

// BAD: Don't use unwrap/expect in library code
let result = some_option.unwrap();  // Never do this!
```

### Unsafe Code Policy

`unsafe` code requires:

1. **Strong justification** - Document why safe alternatives are insufficient
2. **Safety invariants** - Comment explaining why this use is sound
3. **Minimal scope** - Keep unsafe blocks as small as possible
4. **Review by maintainer** - All unsafe code requires explicit approval

```rust
// REQUIRED: Document safety invariants
// SAFETY: We have exclusive access to `buffer` through the mutable reference,
// and `len` is guaranteed to be within bounds by the preceding check.
unsafe {
    std::ptr::copy_nonoverlapping(src.as_ptr(), buffer.as_mut_ptr(), len);
}
```

---

## Testing Requirements

### Test Coverage Expectations

| Change Type | Test Requirement                              |
| ----------- | --------------------------------------------- |
| Bug fix     | Add regression test that fails without fix    |
| New feature | Unit tests + integration test (if applicable) |
| Refactoring | Existing tests must continue to pass          |
| Performance | Add benchmark if claiming improvement         |

### Running Tests

```bash
# Run all tests
cargo test

# Run with all features enabled
cargo test --all-features

# Run specific test
cargo test test_name

# Run tests in a specific module
cargo test module_name::

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel (faster)
cargo nextest run  # requires cargo-nextest
```

### Writing Good Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // GOOD: Descriptive test name following given_when_then pattern
    #[test]
    fn search_with_empty_query_returns_error() {
        let client = SearchClient::new().unwrap();
        let result = client.search("");

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SearchError::EmptyQuery));
    }

    // GOOD: Test edge cases
    #[test]
    fn search_with_unicode_query_succeeds() {
        let client = SearchClient::new().unwrap();
        let result = client.search("cafe").unwrap();

        assert!(!result.is_empty());
    }

    // GOOD: Use proptest for property-based testing
    #[cfg(test)]
    mod property_tests {
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn search_never_panics(query in "\\PC*") {
                let client = SearchClient::new().unwrap();
                let _ = client.search(&query);  // Should not panic
            }
        }
    }
}
```

### Benchmark Requirements

Performance-critical changes should include benchmarks:

```rust
// benches/search_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_search(c: &mut Criterion) {
    let client = SearchClient::new().unwrap();

    c.bench_function("hybrid_search_100_docs", |b| {
        b.iter(|| client.hybrid_search(black_box("test query"), 10))
    });
}

criterion_group!(benches, benchmark_search);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench

# Compare against baseline
cargo bench -- --save-baseline main
cargo bench -- --baseline main
```

---

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/) for clear, parseable commit history.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type       | When to Use                                           |
| ---------- | ----------------------------------------------------- |
| `feat`     | New feature                                           |
| `fix`      | Bug fix                                               |
| `docs`     | Documentation only                                    |
| `style`    | Formatting, missing semicolons (no code change)       |
| `refactor` | Code change that neither fixes a bug nor adds feature |
| `perf`     | Performance improvement                               |
| `test`     | Adding or updating tests                              |
| `build`    | Build system or external dependencies                 |
| `ci`       | CI configuration                                      |
| `chore`    | Other changes (e.g., updating .gitignore)             |

### Scopes (Optional)

Use the module or component name:

- `search`, `retrieval`, `embedding`, `ingestion`
- `thinktool`, `protocol`, `trace`
- `cli`, `api`, `config`
- `docs`, `ci`, `deps`

### Examples

```bash
# Feature
feat(search): add BM25 scoring algorithm

# Bug fix with issue reference
fix(ingestion): handle empty PDF pages

Fixes #123

# Breaking change (note the !)
feat(api)!: change search return type to Result<T, E>

BREAKING CHANGE: search() now returns Result instead of Option.
Callers must handle the error case.

# Documentation
docs(readme): update installation instructions

# Performance with details
perf(embedding): batch vector operations

Reduce embedding latency by 40% by batching API calls.
Before: 150ms average
After: 90ms average

# Multiple scopes
fix(search,retrieval): normalize scores before fusion
```

### Bad Examples (Avoid These)

```bash
# Too vague
fix: fixed bug

# No type
updated search module

# Past tense (use imperative)
feat(search): added new algorithm

# Too long first line (keep under 72 chars)
feat(search): implement the new revolutionary algorithm that will change everything about how we do search
```

---

## Pull Request Process

### Workflow

```
1. Fork       2. Branch      3. Code       4. Test       5. PR
   |             |             |             |             |
   v             v             v             v             v
[GitHub]    [feature/x]    [Changes]    [All Pass]    [Review]
```

### Step-by-Step

1. **Fork** the repository on GitHub

2. **Create a branch** from `main`:

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the code guidelines

4. **Run the quality gates** (mandatory):

   ```bash
   cargo build --release        # Gate 1: Build
   cargo clippy -- -D warnings  # Gate 2: Lint
   cargo fmt --check            # Gate 3: Format
   cargo test --all-features    # Gate 4: Tests
   cargo bench                  # Gate 5: Performance (if applicable)
   ```

5. **Commit** with conventional commit messages

6. **Push** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

### Branch Naming

| Prefix      | Purpose                     | Example                    |
| ----------- | --------------------------- | -------------------------- |
| `feature/`  | New features                | `feature/hybrid-search`    |
| `fix/`      | Bug fixes                   | `fix/chunk-splitting`      |
| `docs/`     | Documentation               | `docs/api-reference`       |
| `perf/`     | Performance improvements    | `perf/vector-optimization` |
| `test/`     | Test additions/improvements | `test/search-edge-cases`   |
| `refactor/` | Code refactoring            | `refactor/error-handling`  |

### PR Checklist

Before submitting, ensure:

- [ ] Code compiles: `cargo build --release`
- [ ] No lint warnings: `cargo clippy -- -D warnings`
- [ ] Code is formatted: `cargo fmt --check`
- [ ] All tests pass: `cargo test`
- [ ] Documentation updated (if applicable)
- [ ] Tests added for new functionality
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes
- [ ] No unrelated changes included

### The 5 Gates of Quality (CONS-009)

All PRs must pass these gates. They are enforced by CI.

| Gate | Command                       | Threshold       | Enforcement |
| ---- | ----------------------------- | --------------- | ----------- |
| 1    | `cargo build --release`       | Exit 0          | BLOCKING    |
| 2    | `cargo clippy -- -D warnings` | 0 warnings      | BLOCKING    |
| 3    | `cargo fmt --check`           | Pass            | BLOCKING    |
| 4    | `cargo test --all-features`   | 100% pass       | BLOCKING    |
| 5    | `cargo bench`                 | < 5% regression | ADVISORY    |

### Review Process

1. **Automated checks** run immediately via GitHub Actions
2. **Maintainer review** typically within 1 week
3. **Address feedback** - update your PR as needed
4. **Approval and merge** - maintainer will merge once approved

### After Your PR is Merged

```bash
# Update your local main
git checkout main
git pull upstream main

# Delete your feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## Issue Guidelines

### Before Opening an Issue

1. **Search existing issues** - your issue may already be reported
2. **Update to latest version** - the bug may already be fixed
3. **Prepare a minimal reproduction** - helps us fix it faster

### Issue Templates

We provide templates to help you create effective issues:

- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) - Report a bug
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml) - Suggest an enhancement
- [Question](.github/ISSUE_TEMPLATE/question.md) - Ask a question

### Bug Report Essentials

Include:

- ReasonKit version (`rk-core --version`)
- Rust version (`rustc --version`)
- Operating system and version
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

### Feature Request Essentials

Include:

- Clear problem statement / use case
- Proposed solution
- Alternatives you have considered
- Willingness to contribute

---

## Getting Help

| Channel                | Best For                  | Link                                                                   |
| ---------------------- | ------------------------- | ---------------------------------------------------------------------- |
| **GitHub Discussions** | Questions, ideas, help    | [Discussions](https://github.com/reasonkit/reasonkit-core/discussions) |
| **GitHub Issues**      | Bugs, feature requests    | [Issues](https://github.com/reasonkit/reasonkit-core/issues)           |
| **Security Issues**    | Vulnerabilities (private) | <security@reasonkit.sh>                                                  |

### Response Times

| Type             | Target Response Time |
| ---------------- | -------------------- |
| Security issues  | < 24 hours           |
| Bug reports      | < 48 hours           |
| Feature requests | < 1 week             |
| Pull requests    | < 1 week             |

---

## Recognition

Contributors are recognized in:

- **Release notes** for significant contributions
- **GitHub contributors page**
- **CONTRIBUTORS.md** (coming soon)

We value every contribution, from typo fixes to major features. Your work matters!

---

## License

By contributing to ReasonKit Core, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

---

## Thank You

Every contribution makes ReasonKit better. Whether you are fixing a typo or implementing a major feature, we appreciate your effort and time.

Questions? Open a [Discussion](https://github.com/reasonkit/reasonkit-core/discussions).

---

*"Designed, Not Dreamed. Turn Prompts into Protocols."*

<https://reasonkit.sh>
