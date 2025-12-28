# Versioning Policy

This document describes the versioning policy for ReasonKit Core.

---

## Semantic Versioning

ReasonKit Core follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Version numbers are formatted as: **MAJOR.MINOR.PATCH**

```
X.Y.Z
│ │ │
│ │ └─ PATCH: Bug fixes (backward compatible)
│ └─── MINOR: New features (backward compatible)
└───── MAJOR: Breaking changes
```

---

## Version Stages

### Pre-1.0 (Current Stage)

```
0.x.y - Development versions
```

During pre-1.0 development:

- **Minor versions** (0.x.0) MAY include breaking changes
- **Patch versions** (0.x.y) SHOULD be backward compatible
- API stability is NOT guaranteed
- Breaking changes are documented in CHANGELOG.md

**Current Version:** 0.1.0

### Post-1.0 (Future)

```
1.0.0+ - Stable versions
```

After 1.0.0 release:

- **Major versions** (x.0.0) - Breaking changes
- **Minor versions** (x.y.0) - New features, backward compatible
- **Patch versions** (x.y.z) - Bug fixes, backward compatible
- Public API stability is guaranteed within major versions

---

## What Constitutes a Breaking Change?

### Breaking Changes (Requires Major Version Bump)

1. **CLI Changes**
   - Removing a command or subcommand
   - Changing required arguments
   - Changing output format in incompatible ways
   - Renaming commands without aliases

2. **Configuration Changes**
   - Removing configuration options
   - Changing configuration file format
   - Changing environment variable names

3. **Protocol Changes**
   - Removing ThinkTools or modules
   - Changing protocol YAML schema incompatibly
   - Altering step execution order in breaking ways

4. **API Changes** (Library Use)
   - Removing public functions/structs
   - Changing function signatures
   - Removing struct fields
   - Changing return types

### Non-Breaking Changes (Minor or Patch)

1. **Additions**
   - New CLI commands or options
   - New ThinkTools or modules
   - New configuration options
   - New profile presets

2. **Improvements**
   - Performance optimizations
   - Better error messages
   - Additional output formats (if existing formats unchanged)
   - New LLM provider support

3. **Bug Fixes**
   - Fixing incorrect behavior
   - Fixing crashes or errors
   - Correcting documentation

---

## Version Bumping Guidelines

### When to Bump PATCH (0.1.x)

- Bug fixes that don't change behavior
- Documentation corrections
- Internal refactoring with no API changes
- Dependency updates (non-breaking)

```bash
# Example: 0.1.0 -> 0.1.1
cargo release patch
```

### When to Bump MINOR (0.x.0)

- New features
- New ThinkTools modules
- New CLI commands
- New profile presets
- Deprecations (feature still works, warning issued)

```bash
# Example: 0.1.0 -> 0.2.0
cargo release minor
```

### When to Bump MAJOR (x.0.0)

- Breaking API changes
- Removed features
- Incompatible configuration changes
- Major architectural changes

```bash
# Example: 0.1.0 -> 1.0.0
cargo release major
```

---

## Release Process

### 1. Pre-Release Checklist

```bash
# Run all quality gates
cargo build --release
cargo clippy -- -D warnings
cargo fmt --check
cargo test --all-features
cargo audit
```

### 2. Update Version

Update `Cargo.toml`:

```toml
[package]
version = "X.Y.Z"
```

### 3. Update Changelog

Move items from `[Unreleased]` to new version section:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature...

### Changed
- Changed behavior...

### Fixed
- Bug fix...
```

### 4. Create Git Tag

```bash
git add -A
git commit -m "chore: release vX.Y.Z"
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin main --tags
```

### 5. Publish

```bash
# Crates.io
cargo publish

# PyPI (if bindings ready)
maturin publish
```

### 6. Post-Release

- Create GitHub Release with release notes
- Update website/documentation
- Announce on relevant channels

---

## Deprecation Policy

### How We Deprecate

1. **Announce** - Document in CHANGELOG.md under "Deprecated"
2. **Warn** - Emit deprecation warnings when deprecated features are used
3. **Maintain** - Keep deprecated features for at least one minor version
4. **Remove** - Remove in next major version

### Deprecation Timeline

```
v0.x.0 - Feature deprecated, warning added
v0.y.0 - Warning continues (minimum 1 minor version)
v1.0.0 - Feature may be removed (major version)
```

### Deprecation Notice Format

```rust
#[deprecated(
    since = "0.2.0",
    note = "Use `new_function` instead. Will be removed in 1.0.0."
)]
pub fn old_function() { ... }
```

---

## Pre-Release Versions

For testing and early access:

```
X.Y.Z-alpha.N  - Early development, unstable
X.Y.Z-beta.N   - Feature complete, testing
X.Y.Z-rc.N     - Release candidate, final testing
```

Example progression:

```
0.2.0-alpha.1  First alpha
0.2.0-alpha.2  Second alpha
0.2.0-beta.1   Feature freeze
0.2.0-rc.1     Release candidate
0.2.0          Final release
```

---

## Build Metadata

Build metadata can be appended for tracking:

```
X.Y.Z+build.N     - Build number
X.Y.Z+git.SHA     - Git commit SHA
X.Y.Z+date.YYMMDD - Build date
```

Build metadata does NOT affect version precedence.

---

## Version in Code

The version is defined in `Cargo.toml` and accessible at runtime:

```rust
// In Rust code
const VERSION: &str = env!("CARGO_PKG_VERSION");

// CLI output
rk-core --version
// Output: rk-core 0.1.0
```

---

## Compatibility Matrix

| reasonkit-core | Rust MSRV | Python Bindings | Status        |
| -------------- | --------- | --------------- | ------------- |
| 0.1.x          | 1.74+     | 3.9+            | Current       |
| 0.2.x          | 1.74+     | 3.9+            | Planned       |
| 1.0.x          | TBD       | TBD             | Future stable |

**MSRV** = Minimum Supported Rust Version

---

## Questions?

- See [CHANGELOG.md](../CHANGELOG.md) for version history
- See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for release steps
- See [PUBLISHING.md](PUBLISHING.md) for crates.io publishing

---

*Last Updated: 2025-12-28*
*Maintained By: ReasonKit Team*
