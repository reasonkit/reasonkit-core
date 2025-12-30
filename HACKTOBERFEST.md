# Hacktoberfest 2025 - Contribute to ReasonKit! 🎃

> **Turn Prompts into Protocols** — Help us make AI reasoning structured, auditable, and reliable.

Welcome to ReasonKit's Hacktoberfest guide! We're excited to have you contribute. This guide will help you make your first (or next) contribution quickly and successfully.

---

## 🎯 What is Hacktoberfest?

[Hacktoberfest](https://hacktoberfest.com/) is a month-long celebration of open source. Contribute 4 pull requests to any participating repository in October, and you can earn a limited-edition T-shirt or plant a tree.

**ReasonKit is participating!** All valid PRs merged during October count toward your Hacktoberfest goals.

---

## 🚀 Quick Start (5 Minutes)

### 1. Fork & Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/reasonkit-core.git
cd reasonkit-core
git remote add upstream https://github.com/reasonkit/reasonkit-core.git
```

### 2. Build & Test

```bash
# Install Rust if needed: https://rustup.rs/
cargo build
cargo test
```

If tests pass, you're ready!

### 3. Pick an Issue

Browse our [Good First Issues](#-good-first-issues) below or check:

- [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue")
- [Documentation improvements](#-documentation-improvements)
- [Quick wins](#-quick-wins-no-issue-needed)

---

## 🎁 Good First Issues

These issues are perfect for Hacktoberfest contributors:

### Documentation (30 min - 2 hours)

| Issue                | Description                       | Difficulty  | Time      |
| -------------------- | --------------------------------- | ----------- | --------- |
| **Fix typos**        | Find and fix typos in docs        | ⭐ Easy     | 15-30 min |
| **Improve examples** | Add more code examples            | ⭐ Easy     | 30-60 min |
| **Add missing docs** | Document undocumented functions   | ⭐⭐ Medium | 1-2 hours |
| **Translation**      | Translate docs to other languages | ⭐⭐ Medium | 2-4 hours |

**Where to find:**

- `docs/` directory
- `README.md`
- Code comments (`///` doc comments)

### Code Quality (1-3 hours)

| Issue                      | Description                      | Difficulty  | Time      |
| -------------------------- | -------------------------------- | ----------- | --------- |
| **Add tests**              | Write tests for untested code    | ⭐⭐ Medium | 1-2 hours |
| **Improve error messages** | Make errors more helpful         | ⭐ Easy     | 30-60 min |
| **Code cleanup**           | Remove unused imports, dead code | ⭐ Easy     | 15-30 min |
| **Add doc comments**       | Document public functions        | ⭐ Easy     | 30-60 min |

**Where to find:**

- Look for `pub fn` without `///` doc comments
- Check `cargo clippy` warnings
- Find functions with no tests

### CLI Improvements (1-2 hours)

| Issue                   | Description                    | Difficulty  | Time      |
| ----------------------- | ------------------------------ | ----------- | --------- |
| **Add command aliases** | `t=think`, `w=web`, `v=verify` | ⭐⭐ Medium | 1-2 hours |
| **Improve help text**   | Make CLI help more descriptive | ⭐ Easy     | 30-60 min |
| **Add examples**        | Add usage examples to CLI help | ⭐ Easy     | 30-60 min |

**Where to find:**

- `src/bin/rk-core/main.rs`
- CLI command definitions

### Examples & Tutorials (2-4 hours)

| Issue                       | Description                        | Difficulty      | Time      |
| --------------------------- | ---------------------------------- | --------------- | --------- |
| **Add example script**      | Create bash script using ReasonKit | ⭐⭐ Medium     | 2-3 hours |
| **Write tutorial**          | Step-by-step guide for a use case  | ⭐⭐⭐ Advanced | 3-4 hours |
| **Add integration example** | Show ReasonKit + other tools       | ⭐⭐⭐ Advanced | 3-4 hours |

**Where to find:**

- `scripts/examples/` directory
- `docs/CLI_WORKFLOW_EXAMPLES.md`

---

## ⚡ Quick Wins (No Issue Needed)

These contributions are always welcome and don't require opening an issue first:

### 1. Fix a Typo (5-15 minutes)

```bash
# Find typos
grep -r "teh\|recieve\|seperate" docs/ README.md

# Fix and commit
git checkout -b docs/fix-typo
# Edit file
git commit -m "docs: fix typo in README.md"
git push origin docs/fix-typo
```

### 2. Improve Error Messages (15-30 minutes)

Find generic error messages and make them more helpful:

```rust
// Before
return Err("Error");

// After
return Err(anyhow!("Failed to parse protocol: expected YAML format, got {:?}", input));
```

### 3. Add Missing Doc Comments (15-30 minutes)

Find public functions without documentation:

```bash
# Find undocumented public functions
grep -r "pub fn" src/ | grep -v "///"
```

Add doc comments:

```rust
/// Parses a protocol definition from YAML.
///
/// # Arguments
/// * `yaml` - YAML string containing protocol definition
///
/// # Returns
/// Parsed `Protocol` struct or error if invalid
pub fn parse_protocol(yaml: &str) -> Result<Protocol> {
    // ...
}
```

### 4. Add Tests (30-60 minutes)

Find untested code paths and add tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_case() {
        // Your test here
    }
}
```

### 5. Update Outdated Comments (10-20 minutes)

Find comments that don't match the code:

```bash
# Look for TODO, FIXME, or outdated comments
grep -r "TODO\|FIXME\|XXX" src/
```

---

## 📝 Making Your Contribution

### Step-by-Step Workflow

1. **Pick an issue or quick win** from above
2. **Create a branch:**
   ```bash
   git checkout -b fix/your-change-description
   ```
3. **Make your changes**
4. **Format and lint:**
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   ```
5. **Run tests:**
   ```bash
   cargo test
   ```
6. **Commit:**
   ```bash
   git commit -m "docs: fix typo in README.md"
   ```
7. **Push and create PR:**
   ```bash
   git push origin fix/your-change-description
   ```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

Examples:
docs(readme): fix typo in installation section
fix(cli): improve error message for invalid protocol
test(retrieval): add edge case test for empty query
```

**Types:** `docs`, `fix`, `feat`, `test`, `perf`, `refactor`, `style`

---

## ✅ PR Checklist

Before submitting your PR:

- [ ] **Builds:** `cargo build --release` passes
- [ ] **Lints:** `cargo clippy -- -D warnings` has 0 errors
- [ ] **Formatted:** `cargo fmt --check` passes
- [ ] **Tests:** `cargo test` passes (or `cargo test` if `--all-features` fails)
- [ ] **Documentation:** Updated if needed
- [ ] **Commit message:** Follows conventional commits format
- [ ] **PR description:** Explains what and why

---

## 🎓 Learning Resources

### New to Rust?

- [Rust Book](https://doc.rust-lang.org/book/) - Official Rust tutorial
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn by example
- [Rustlings](https://github.com/rust-lang/rustlings) - Interactive exercises

### New to ReasonKit?

- [Contributor Quick Start](docs/CONTRIBUTOR_QUICKSTART.md) - 30-minute guide
- [Architecture Overview](ARCHITECTURE.md) - System design
- [ThinkTools Guide](docs/THINKTOOLS_V2_GUIDE.md) - How ThinkTools work

### New to Open Source?

- [First Contributions](https://github.com/firstcontributions/first-contributions) - General guide
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/) - GitHub guide

---

## 🏆 Recognition

All contributors are recognized:

- **GitHub Contributors** - Listed on our [Contributors page](https://github.com/reasonkit/reasonkit-core/graphs/contributors)
- **Release Notes** - Significant contributions mentioned in release notes
- **Hacktoberfest** - Valid PRs count toward your Hacktoberfest goals

---

## 🆘 Getting Help

### Stuck?

1. **Check existing issues** - Your question may already be answered
2. **Ask in Discussions** - [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions)
3. **Open an issue** - Use the "Question" template

### Response Times

- **Issues:** Within 48 hours
- **PRs:** Within 1 week

---

## 🎯 Hacktoberfest Rules

### Valid Contributions

✅ **Valid:**

- Bug fixes
- Documentation improvements
- Code quality improvements
- Tests
- Examples and tutorials
- Translation work

❌ **Invalid:**

- Spam PRs (e.g., adding your name to a list)
- PRs that don't follow our guidelines
- PRs that don't pass quality gates
- Duplicate PRs

### Quality Standards

All PRs must:

1. Pass all 5 quality gates (see [CONTRIBUTING.md](CONTRIBUTING.md))
2. Follow code style guidelines
3. Include tests if adding features
4. Update documentation if needed

**We review all PRs carefully** - low-quality PRs will be marked as `invalid` for Hacktoberfest.

---

## 📚 Project Structure

```
reasonkit-core/
├── src/                    # Source code
│   ├── bin/                # CLI binaries
│   ├── thinktool/          # ThinkTools (core reasoning)
│   ├── retrieval/          # Search and retrieval
│   └── ingestion/          # Document processing
├── docs/                   # Documentation
├── tests/                  # Integration tests
├── benches/                # Performance benchmarks
└── scripts/                # Utility scripts
```

**Key files for contributors:**

- `CONTRIBUTING.md` - Full contribution guide
- `docs/CONTRIBUTOR_QUICKSTART.md` - Quick start guide
- `ARCHITECTURE.md` - System architecture
- `README.md` - Project overview

---

## 🌟 Contribution Ideas by Skill Level

### Beginner (No Rust Experience)

- Fix typos in documentation
- Improve error messages
- Add missing doc comments
- Update outdated comments
- Improve README examples

### Intermediate (Some Rust Experience)

- Add tests for untested code
- Implement CLI improvements
- Add example scripts
- Improve error handling
- Refactor code for clarity

### Advanced (Rust Expert)

- Implement new features
- Optimize performance
- Add new ThinkTool modules
- Integrate new LLM providers
- Improve architecture

---

## 🎉 Ready to Contribute?

1. **Fork** the repository
2. **Pick** an issue or quick win
3. **Make** your changes
4. **Submit** a PR
5. **Celebrate** your contribution!

**Questions?** Open a [Discussion](https://github.com/reasonkit/reasonkit-core/discussions) or join our [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions).

---

**Happy Hacking! 🎃**

_Turn Prompts into Protocols. Make AI reasoning structured, auditable, and reliable._

---

**Related:**

- [Contributing Guide](CONTRIBUTING.md) - Complete guide
- [Contributor Quick Start](docs/CONTRIBUTOR_QUICKSTART.md) - 30-minute guide
- [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue") - GitHub issues
