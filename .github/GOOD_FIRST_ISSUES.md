# Good First Issues

> Curated list of beginner-friendly issues perfect for first-time contributors

This document lists issues that are well-suited for new contributors. These issues are:

- **Well-scoped** - Clear requirements and expected outcomes
- **Beginner-friendly** - Don't require deep knowledge of the codebase
- **Self-contained** - Can be completed independently
- **Documented** - Have clear instructions

---

## 🏷️ Issue Labels

Look for these labels on GitHub:

- `good first issue` - Perfect for first-time contributors
- `documentation` - Documentation improvements
- `help wanted` - Community help needed
- `easy` - Quick wins (1-3 hours)

**Browse on GitHub:** [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue")

---

## 📝 Documentation Issues

### D-001: Fix Typos in Documentation

**Difficulty:** ⭐ Easy  
**Time:** 15-30 minutes  
**Labels:** `documentation`, `good first issue`

**Description:**
Find and fix typos, grammatical errors, and spelling mistakes in documentation files.

**Files to check:**

- `README.md`
- `docs/*.md`
- Code comments (`///` doc comments)

**How to contribute:**

1. Search for common typos: `grep -r "teh\|recieve\|seperate" docs/ README.md`
2. Fix the typos
3. Submit PR with message: `docs: fix typos in [filename]`

---

### D-002: Add Missing Code Examples

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `documentation`, `good first issue`

**Description:**
Add code examples to documentation that currently lacks them.

**Where to find:**

- Functions without examples in doc comments
- API documentation pages
- Tutorial sections

**Example:**

````rust
/// Parses a protocol from YAML.
///
/// # Example
/// ```rust
/// use reasonkit_core::thinktool::Protocol;
///
/// let yaml = r#"
/// id: test-protocol
/// name: Test Protocol
/// "#;
///
/// let protocol = Protocol::from_yaml(yaml)?;
/// ```
pub fn from_yaml(yaml: &str) -> Result<Protocol> {
    // ...
}
````

---

### D-003: Improve Error Messages

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `documentation`, `good first issue`

**Description:**
Find generic error messages and make them more helpful and actionable.

**Where to find:**

- `src/**/*.rs` - Look for `Err("Error")` or similar
- CLI error messages
- Validation errors

**Example:**

```rust
// Before
return Err(anyhow!("Error"));

// After
return Err(anyhow!(
    "Failed to parse protocol: expected YAML format, got {:?}. \
     See https://docs.reasonkit.sh/protocols for format specification.",
    input
));
```

---

### D-004: Add Missing Doc Comments

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `documentation`, `good first issue`

**Description:**
Find public functions without documentation and add doc comments.

**How to find:**

```bash
# Find undocumented public functions
grep -r "pub fn" src/ | grep -v "///"
```

**Template:**

````rust
/// Brief description of what the function does.
///
/// More detailed explanation if needed.
///
/// # Arguments
/// * `param1` - Description of parameter
///
/// # Returns
/// Description of return value
///
/// # Errors
/// When this function returns an error
///
/// # Example
/// ```rust
/// // Example usage
/// ```
pub fn function_name(param1: Type) -> Result<ReturnType> {
    // ...
}
````

---

## 🧪 Testing Issues

### T-001: Add Edge Case Tests

**Difficulty:** ⭐⭐ Medium  
**Time:** 1-2 hours  
**Labels:** `testing`, `good first issue`

**Description:**
Add tests for edge cases that aren't currently covered.

**Where to find:**

- Functions with minimal test coverage
- Error paths that aren't tested
- Boundary conditions

**Example:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = parse_protocol("");
        assert!(result.is_err());
    }

    #[test]
    fn test_very_large_input() {
        let large_input = "x".repeat(1_000_000);
        let result = parse_protocol(&large_input);
        // Test behavior
    }
}
```

---

### T-002: Improve Test Coverage

**Difficulty:** ⭐⭐ Medium  
**Time:** 1-2 hours  
**Labels:** `testing`, `good first issue`

**Description:**
Identify modules with low test coverage and add tests.

**How to check coverage:**

```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
```

**Focus areas:**

- Error handling paths
- Edge cases
- Integration scenarios

---

## 🛠️ Code Quality Issues

### C-001: Remove Unused Imports

**Difficulty:** ⭐ Easy  
**Time:** 15-30 minutes  
**Labels:** `code quality`, `good first issue`

**Description:**
Find and remove unused imports flagged by clippy.

**How to find:**

```bash
cargo clippy -- -D warnings 2>&1 | grep "unused_import"
```

**Fix:**
Remove the unused import or add `#[allow(unused_imports)]` if it's needed for documentation.

---

### C-002: Fix Clippy Warnings

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `code quality`, `good first issue`

**Description:**
Fix clippy warnings to improve code quality.

**How to find:**

```bash
cargo clippy -- -D warnings
```

**Common fixes:**

- Use `is_empty()` instead of `len() == 0`
- Use `if let` instead of `match` for single patterns
- Remove unnecessary `clone()` calls
- Use `&str` instead of `&String`

---

### C-003: Improve Code Comments

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `code quality`, `good first issue`

**Description:**
Update outdated comments, add missing explanations, or clarify complex logic.

**Where to find:**

- Complex algorithms without comments
- TODO/FIXME comments
- Outdated comments that don't match code

---

## 🖥️ CLI Improvements

### CLI-001: Add Command Aliases

**Difficulty:** ⭐⭐ Medium  
**Time:** 1-2 hours  
**Labels:** `cli`, `good first issue`

**Description:**
Add command aliases for common commands:

- `t` → `think`
- `w` → `web`
- `v` → `verify`

**Where to implement:**

- `src/main.rs`
- CLI argument parsing

---

### CLI-002: Improve Help Text

**Difficulty:** ⭐ Easy  
**Time:** 30-60 minutes  
**Labels:** `cli`, `good first issue`

**Description:**
Make CLI help text more descriptive and include examples.

**Where to find:**

- `src/main.rs`
- Command descriptions
- Option descriptions

**Example:**

```rust
// Before
.about("Execute ThinkTool")

// After
.about("Execute a ThinkTool protocol for structured reasoning")
.long_about(
    "Execute a ThinkTool protocol to analyze a query with structured reasoning.\n\n\
     Examples:\n\
     rk think \"Should I take this job?\" --profile balanced\n\
     rk think \"Is this email a phishing attempt?\" --profile quick"
)
```

---

## 📚 Examples & Tutorials

### E-001: Add Example Scripts

**Difficulty:** ⭐⭐ Medium  
**Time:** 2-3 hours  
**Labels:** `examples`, `good first issue`

**Description:**
Create bash scripts demonstrating ReasonKit usage for common scenarios.

**Where to add:**

- `scripts/examples/` directory

**Example scenarios:**

- Career decision analysis
- Code review automation
- Research paper analysis
- Business strategy evaluation

**Template:**

```bash
#!/usr/bin/env bash
# Example: [Scenario Name]
# Description: [What this script demonstrates]

set -euo pipefail

echo "Running ReasonKit analysis..."

rk think \
    "Your question here" \
    --profile balanced \
    --output json > result.json

echo "Analysis complete! Results saved to result.json"
```

---

### E-002: Write Tutorial

**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** 3-4 hours  
**Labels:** `examples`, `documentation`

**Description:**
Write a step-by-step tutorial for a specific use case.

**Where to add:**

- `docs/tutorials/` directory

**Tutorial ideas:**

- "Building a Decision Support System with ReasonKit"
- "Integrating ReasonKit into a Python Application"
- "Using ReasonKit for Code Review Automation"

---

## 🎨 UI/UX Improvements

### U-001: Improve CLI Output Formatting

**Difficulty:** ⭐⭐ Medium  
**Time:** 1-2 hours  
**Labels:** `cli`, `ux`, `good first issue`

**Description:**
Improve the visual formatting of CLI output for better readability.

**Areas to improve:**

- Progress indicators
- Error message formatting
- Result display
- Color coding (with proper terminal support)

---

## 🔍 How to Find More Issues

### On GitHub

1. **Filter by labels:**
   - [Good First Issues](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"good+first+issue")
   - [Documentation](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"documentation")
   - [Help Wanted](https://github.com/reasonkit/reasonkit-core/issues?q=is:open+label:"help+wanted")

2. **Search by keyword:**
   - `is:issue is:open "good first issue"`
   - `is:issue is:open "documentation"`

### In the Codebase

1. **Run clippy:**

   ```bash
   cargo clippy -- -D warnings
   ```

   Fix any warnings you find.

2. **Check test coverage:**

   ```bash
   cargo tarpaulin
   ```

   Add tests for uncovered code.

3. **Search for TODOs:**

   ```bash
   grep -r "TODO\|FIXME\|XXX" src/
   ```

---

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/reasonkit-core.git
   cd reasonkit-core
   ```

3. **Pick an issue** from this list or GitHub
4. **Create a branch:**

   ```bash
   git checkout -b fix/issue-description
   ```

5. **Make your changes**
6. **Test:**

   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

7. **Commit and push:**

   ```bash
   git commit -m "fix: [description]"
   git push origin fix/issue-description
   ```

8. **Open a PR** on GitHub

---

## 📖 Resources

- [Contributing Guide](CONTRIBUTING.md) - Complete contribution guidelines
- [Contributor Quick Start](docs/getting-started/CONTRIBUTOR_QUICKSTART.md) - 30-minute guide
- [Architecture](ARCHITECTURE.md) - System design overview

---

## 🆘 Need Help?

- **GitHub Discussions** - [Ask questions](https://github.com/reasonkit/reasonkit-core/discussions)
- **Open an issue** - Use the "Question" template

---

**Happy Contributing! 🎉**

*Turn Prompts into Protocols. Make AI reasoning structured, auditable, and reliable.*
