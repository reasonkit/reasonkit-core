# Publishing Checklist for reasonkit-core

This document provides a comprehensive checklist for publishing `reasonkit-core` to crates.io.

## Pre-Publishing Verification

### 1. Package Metadata (Cargo.toml)

- [x] **name**: `reasonkit-core` (unique on crates.io)
- [x] **version**: `1.0.0` (follows semantic versioning)
- [x] **edition**: `2021` (current Rust edition)
- [x] **rust-version**: `1.70` (minimum supported Rust version)
- [x] **authors**: `ReasonKit Team <hello@reasonkit.sh>`
- [x] **description**: Clear, concise description (< 200 chars)
- [x] **license**: `Apache-2.0` (OSI-approved)
- [x] **repository**: GitHub URL
- [x] **homepage**: `https://reasonkit.sh`
- [x] **documentation**: `https://docs.rs/reasonkit-core`
- [x] **readme**: `README.md` (included in package)
- [x] **keywords**: 5 relevant keywords for discoverability
- [x] **categories**: 3 appropriate categories from crates.io list
- [x] **exclude**: Large data directories and unnecessary files excluded

### 2. License & Legal

- [x] **LICENSE file**: Apache 2.0 license text present in repository root
- [ ] **Copyright notices**: All source files have appropriate headers
- [ ] **Third-party licenses**: All dependencies are Apache 2.0 compatible
- [ ] **No proprietary code**: Verify no proprietary or confidential code included

### 3. Documentation

- [x] **README.md**: Comprehensive introduction with examples
- [ ] **API documentation**: All public APIs have rustdoc comments
- [ ] **Module docs**: All modules have module-level documentation
- [ ] **Examples**: At least 2-3 working examples in `examples/` directory
- [ ] **Architecture docs**: ARCHITECTURE.md included (excluded from package but on GitHub)

### 4. Code Quality

- [ ] **Build passes**: `cargo build --release` succeeds
- [ ] **Tests pass**: `cargo test --all-features` succeeds (100% pass rate)
- [ ] **Clippy clean**: `cargo clippy -- -D warnings` reports 0 errors
- [ ] **Format check**: `cargo fmt --check` passes
- [ ] **No warnings**: Build produces no warnings on stable Rust
- [ ] **Benchmarks work**: `cargo bench` runs successfully

### 5. Dependencies

- [x] **Crates.io compatible**: All dependencies available on crates.io
- [ ] **Version constraints**: All version constraints are appropriate (not over-constrained)
- [ ] **No path dependencies**: No local path dependencies
- [ ] **No git dependencies**: No git-based dependencies
- [ ] **Optional features**: All optional dependencies properly gated by features

### 6. Package Size & Content

- [ ] **Package size check**: Run `cargo package --list` to verify contents
- [ ] **Size under 10MB**: Package size reasonable (exclude data/ to reduce from 537MB)
- [ ] **No secrets**: No API keys, passwords, or credentials in source
- [ ] **No binaries**: No compiled binaries checked in
- [ ] **Exclude list**: Verify `.github/`, `data/`, `benchmarks/`, etc. are excluded

### 7. Features & Functionality

- [ ] **Default feature works**: `cargo build` with default features succeeds
- [ ] **All features work**: `cargo build --all-features` succeeds
- [ ] **Each feature isolated**: `cargo build --features <feature>` for each feature
- [ ] **No feature conflicts**: No incompatible feature combinations
- [ ] **Binary works**: `./target/release/rk-core --help` shows usage

### 8. Cross-Platform Support

- [ ] **Linux builds**: Test on Linux (glibc and musl if possible)
- [ ] **macOS builds**: Test on macOS (Intel and ARM64 if possible)
- [ ] **Windows builds**: Test on Windows (x86_64)
- [ ] **CI passes**: All GitHub Actions workflows pass

## Publishing Commands

### Dry Run (Test Packaging)

```bash
# Clean build to ensure reproducibility
cargo clean

# Create package without uploading
cargo package --allow-dirty

# Verify package builds correctly
cargo publish --dry-run

# Check package contents
cargo package --list | less

# Check package size
ls -lh target/package/reasonkit-core-1.0.0.crate
```

### Pre-Publish Checks

```bash
# 1. Run all quality gates
cargo build --release
cargo clippy -- -D warnings
cargo fmt --check
cargo test --all-features
cargo bench --no-run

# 2. Verify documentation builds
cargo doc --all-features --no-deps

# 3. Test installation from local package
cargo install --path .

# 4. Verify binary works
rk-core --version
rk-core --help
```

### Version Tagging (Before Publishing)

```bash
# Create annotated tag for version
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag to remote
git push origin v1.0.0
```

### Actual Publishing

```bash
# Login to crates.io (one-time setup)
cargo login

# Publish to crates.io
cargo publish

# Verify publication
cargo search reasonkit-core
```

### Post-Publishing Verification

```bash
# Wait a few minutes for crates.io to index

# Test installation from crates.io
cargo install reasonkit-core

# Verify installed binary
rk-core --version

# Check docs.rs built correctly
# Visit: https://docs.rs/reasonkit-core

# Check crates.io page
# Visit: https://crates.io/crates/reasonkit-core
```

## cargo-release Integration

For automated version management and publishing, consider using `cargo-release`:

```toml
# Add to Cargo.toml
[package.metadata.release]
sign-commit = false
sign-tag = false
pre-release-commit-message = "chore: release {{version}}"
post-release-commit-message = "chore: bump version to {{next_version}}"
tag-message = "Release {{version}}"
tag-name = "v{{version}}"
tag-prefix = "v"
pre-release-replacements = [
    { file = "README.md", search = "reasonkit-core = \"[^\"]+\"", replace = "reasonkit-core = \"{{version}}\"" },
    { file = "CHANGELOG.md", search = "## \\[Unreleased\\]", replace = "## [Unreleased]\n\n## [{{version}}] - {{date}}" },
]
```

Then use:

```bash
# Patch version (1.0.0 -> 0.1.1)
cargo release patch

# Minor version (1.0.0 -> 0.2.0)
cargo release minor

# Major version (1.0.0 -> 2.0.0)
cargo release major
```

## Troubleshooting

### Package Too Large

If package exceeds crates.io limit (10MB):

1. Check what's included: `cargo package --list`
2. Add more patterns to `exclude` in Cargo.toml
3. Remove or compress large files
4. Consider splitting into multiple crates

### Documentation Warnings

Fix all rustdoc warnings:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

### Dependency Conflicts

If dependencies conflict:

```bash
# Update Cargo.lock
cargo update

# Check for duplicates
cargo tree --duplicates

# Resolve version conflicts in Cargo.toml
```

### Build Failures on docs.rs

Test docs.rs build locally:

```bash
# Simulate docs.rs environment
cargo +nightly doc --all-features --no-deps

# Check for conditional compilation issues
cargo doc --all-features --no-deps --target x86_64-unknown-linux-gnu
```

## Security Considerations

### Before Publishing

- [ ] Run security audit: `cargo audit`
- [ ] Check for known vulnerabilities: `cargo deny check`
- [ ] Review all dependencies for security issues
- [ ] Ensure no hardcoded secrets or API keys
- [ ] Verify privacy compliance (GDPR, etc.)

### Security Scanning

```bash
# Install cargo-audit
cargo install cargo-audit

# Run audit
cargo audit

# Install cargo-deny
cargo install cargo-deny

# Check licenses and security
cargo deny check
```

## Version Number Guidelines

Follow Semantic Versioning (SemVer):

- **1.0.0**: Initial development release
- **0.x.y**: Pre-1.0 versions (breaking changes allowed in minor)
- **1.0.0**: First stable release (API stability commitment)
- **x.y.z**: Major.Minor.Patch
  - **Major**: Breaking changes
  - **Minor**: New features (backward compatible)
  - **Patch**: Bug fixes (backward compatible)

## Release Notes Template

When publishing, prepare release notes:

````markdown
# reasonkit-core v1.0.0

## Features

- Hybrid search (BM25 + vector embeddings)
- Multi-format document ingestion (PDF, Markdown, HTML, JSON)
- Qdrant vector database integration
- Tantivy full-text search
- ThinkTools cognitive modules (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty)
- CLI with rich commands (ingest, query, stats, serve)

## Performance

- Search latency: <5ms for hybrid queries
- Ingestion throughput: ~100 docs/sec
- Memory footprint: <500MB for 10K documents

## Installation

```bash
cargo install reasonkit-core
```
````

## Documentation

- Docs: https://docs.rs/reasonkit-core
- Repository: https://github.com/reasonkit/reasonkit-core
- Website: https://reasonkit.sh

## Known Limitations

- Requires Qdrant server for vector storage (embedded mode coming soon)
- Large dependency footprint (~200MB total)
- Limited to English language processing

## Next Release

Planned for v0.2.0:

- Embedded Qdrant mode
- RAPTOR hierarchical retrieval
- Additional language support

```

## Checklist Summary

Before running `cargo publish`:

1. ✅ All quality gates pass (build, lint, format, test, bench)
2. ✅ Documentation complete and builds without warnings
3. ✅ LICENSE file present and correct
4. ✅ README.md is comprehensive
5. ✅ All dependencies are crates.io compatible
6. ✅ Package size is reasonable (<10MB)
7. ✅ No secrets or credentials in source
8. ✅ Version number follows SemVer
9. ✅ Git tag created for version
10. ✅ Security audit passes
11. ✅ Cross-platform builds succeed
12. ✅ Release notes prepared

## Support & Maintenance

After publishing:

1. Monitor issues on GitHub
2. Respond to user feedback
3. Maintain backward compatibility
4. Follow deprecation policy for breaking changes
5. Keep dependencies updated
6. Address security vulnerabilities promptly
7. Publish regular updates (monthly or as-needed)

---

**Status**: Ready for publishing after addressing code quality issues (warnings, missing docs)

**Last Updated**: 2025-12-23

**Maintained By**: ReasonKit Team
```
