# Contributor Quick Start

**Goal: Your first PR in 30 minutes or less.**

Welcome! This guide gets you from zero to merged PR as fast as possible.

---

## 5-Minute Setup

### Prerequisites

- Rust 1.74+ ([install rustup](https://rustup.rs/))
- Git 2.30+

### Clone and Build

```bash
# 1. Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/reasonkit-core.git
cd reasonkit-core

# 2. Add upstream
git remote add upstream https://github.com/reasonkit/reasonkit-core.git

# 3. Build (takes ~2 min first time)
cargo build

# 4. Verify tests pass
cargo test
```

If step 4 passes, you are ready to contribute.

---

## Find Your First Issue

Look for these labels on our [Issues page](https://github.com/reasonkit/reasonkit-core/issues):

| Label              | Time Estimate   | Best For                |
| ------------------ | --------------- | ----------------------- |
| `good first issue` | 1-2 hours       | First-time contributors |
| `documentation`    | 30 min - 1 hour | Learning the codebase   |
| `easy`             | 1-3 hours       | Quick wins              |

**Quick link:** [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:issue+is:open+label:%22good+first+issue%22)

### Quick Wins (No Issue Required)

These are always welcome without opening an issue first:

1. **Fix a typo** - in docs, comments, or error messages
2. **Improve error messages** - make them clearer and more helpful
3. **Add missing doc comments** - look for `pub fn` without `///`
4. **Add a test** - find untested code paths
5. **Update outdated comments** - code changed but comments did not

---

## Make Your Change

```bash
# Create a branch
git checkout -b fix/your-change-description

# Make your changes...
# (use your favorite editor)

# Format your code
cargo fmt

# Check for issues
cargo clippy -- -D warnings

# Run tests
cargo test
```

---

## Run Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output visible
cargo test -- --nocapture

# Just doc tests
cargo test --doc
```

---

## Submit Your PR

### 1. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
git add .
git commit -m "fix(docs): correct typo in README"
```

**Common prefixes:**

- `fix:` - Bug fix
- `feat:` - New feature
- `docs:` - Documentation only
- `test:` - Adding tests
- `refactor:` - Code change that is not fix/feat

### 2. Push

```bash
git push origin fix/your-change-description
```

### 3. Open PR on GitHub

- Go to your fork on GitHub
- Click "Compare & pull request"
- Fill in the template
- Submit

### PR Checklist

Before submitting, verify:

- [ ] `cargo build` passes
- [ ] `cargo clippy -- -D warnings` has no errors
- [ ] `cargo fmt --check` passes
- [ ] `cargo test` passes
- [ ] You linked related issues (use `Fixes #123`)

---

## Coding Style Quick Reference

### Naming

| Type                | Convention             | Example              |
| ------------------- | ---------------------- | -------------------- |
| Functions/variables | `snake_case`           | `process_document()` |
| Types/traits        | `PascalCase`           | `SearchResult`       |
| Constants           | `SCREAMING_SNAKE_CASE` | `MAX_CHUNK_SIZE`     |

### Error Handling

```rust
use anyhow::{Context, Result};

pub fn load_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .context("Failed to read file")
}
```

### Document Public APIs

```rust
/// Brief description of what this does.
///
/// # Arguments
///
/// * `query` - The search query
///
/// # Returns
///
/// Description of return value.
pub fn search(query: &str) -> Result<Vec<Hit>> {
    // ...
}
```

### Auto-format

```bash
cargo fmt      # Format code
cargo clippy   # Lint code
```

---

## Where to Ask Questions

| Channel            | Best For                      | Link                                                                   |
| ------------------ | ----------------------------- | ---------------------------------------------------------------------- |
| GitHub Issues      | Bug reports, feature requests | [Issues](https://github.com/reasonkit/reasonkit-core/issues)           |
| GitHub Discussions | Questions, ideas, help        | [Discussions](https://github.com/reasonkit/reasonkit-core/discussions) |
| Discord            | Real-time chat                | [Join Discord](https://discord.gg/reasonkit)                           |

**Response times:** Issues within 48 hours, PRs within 1 week.

---

## Quick Reference Commands

```bash
# Setup
cargo build              # Build project
cargo test               # Run tests

# Before committing
cargo fmt                # Format code
cargo clippy -- -D warnings  # Lint

# Useful
cargo doc --open         # View API docs
cargo bench              # Run benchmarks
```

---

## Project Structure (Key Files)

```
reasonkit-core/
├── src/
│   ├── lib.rs           # Library root
│   ├── main.rs          # CLI entry point
│   ├── thinktool/       # ThinkTools (core reasoning)
│   ├── retrieval/       # Search and retrieval
│   └── ingestion/       # Document processing
├── tests/               # Integration tests
├── benches/             # Performance benchmarks
└── docs/                # Documentation (you are here)
```

---

## Common First PRs

### 1. Documentation Fix

```bash
git checkout -b docs/fix-typo
# Edit the file
cargo fmt
git add .
git commit -m "docs: fix typo in ARCHITECTURE.md"
git push origin docs/fix-typo
```

### 2. Add a Test

```bash
git checkout -b test/add-edge-case
# Add test in src/module.rs or tests/
cargo test test_your_new_test
git add .
git commit -m "test(module): add edge case for empty input"
git push origin test/add-edge-case
```

### 3. Improve Error Message

```bash
git checkout -b fix/better-error-msg
# Find a generic error, make it specific
cargo test
git add .
git commit -m "fix(search): improve error message for empty query"
git push origin fix/better-error-msg
```

---

## You Are Ready

1. Pick an issue or quick win
2. Make your change
3. Run `cargo fmt && cargo clippy -- -D warnings && cargo test`
4. Push and open a PR

**Questions?** Open a [Discussion](https://github.com/reasonkit/reasonkit-core/discussions) - we are happy to help!

---

## Full Guide

For comprehensive contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

_Thank you for contributing to ReasonKit!_

*https://reasonkit.sh*
