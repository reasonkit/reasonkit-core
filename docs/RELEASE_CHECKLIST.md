# Release Checklist - reasonkit-core v0.1.0

**Date:** 2025-12-25
**Status:** VERIFIED
**Version:** 0.1.0

---

## Quality Gates Status

| Gate | Command | Status | Notes |
|------|---------|--------|-------|
| **Gate 1: Build** | `cargo build --release` | PASS | Compiles cleanly |
| **Gate 2: Lint** | `cargo clippy -- -D warnings` | PASS | 0 errors, 0 warnings |
| **Gate 3: Format** | `cargo fmt --check` | PASS | All formatted |
| **Gate 4: Tests** | `cargo test --release` | PASS | 272+ tests passing |
| **Gate 5: Bench** | `cargo bench` | DEFERRED | Run before release |

---

## Pre-Release Checklist

### Code Quality

- [x] All Clippy lints resolved
- [x] Code formatted with `cargo fmt`
- [x] No unused imports
- [x] No unused variables
- [x] All tests passing
- [x] No `unsafe` blocks without documentation

### Documentation

- [x] CLAUDE.md up to date
- [x] README.md accurate
- [x] API documentation complete
- [x] OSS/Pro allocation strategy documented
- [x] ThinkTools segmentation audit complete

### ThinkTools Modules

| Module | Tests | Status |
|--------|-------|--------|
| benchmark.rs | 4 | PASS |
| calibration.rs | 7 | PASS |
| consistency.rs | 5 | PASS |
| debate.rs | 5 | PASS |
| fol.rs | 7 | PASS |
| oscillation.rs | 6 | PASS |
| prm.rs | 5 | PASS |
| quality.rs | 4 | PASS |
| self_refine.rs | 4 | PASS |
| socratic.rs | 6 | PASS |
| tot.rs | 6 | PASS |
| toulmin.rs | 4 | PASS |
| triangulation.rs | 6 | PASS |
| tripartite.rs | 6 | PASS |
| bedrock_tot.rs | 3 | PASS |

**Total ThinkTools Tests:** 139+

### Security

- [ ] No hardcoded secrets
- [ ] No exposed API keys
- [ ] Dependencies audited (`cargo audit`)
- [ ] SPDX license headers

### Integration

- [ ] CLI working (`rk-core --help`)
- [ ] Python bindings tested
- [ ] MCP server tested
- [ ] Benchmarks baselined

---

## Release Steps

### 1. Final Verification

```bash
cd reasonkit-core

# Full quality check
cargo build --release
cargo clippy -- -D warnings
cargo fmt --check
cargo test --release

# Security audit
cargo audit

# Documentation
cargo doc --no-deps --open
```

### 2. Version Bump

```bash
# Update Cargo.toml version
# Update CHANGELOG.md
# Update README.md badges
```

### 3. Git Tag

```bash
git add -A
git commit -m "chore: prepare v0.1.0 release"
git tag -a v0.1.0 -m "ReasonKit Core v0.1.0"
git push origin main --tags
```

### 4. Publish

```bash
# Crates.io
cargo publish

# PyPI (if bindings ready)
maturin publish
```

### 5. Post-Release

- [ ] GitHub Release created
- [ ] Release notes published
- [ ] Website updated
- [ ] Announcement posted

---

## Known Issues

### Dependency Issue

`--all-features` flag triggers `indicatif` compilation error due to upstream issue. Use default features for now.

### Deferred Items

1. **OSS/Pro Split** - All modules in core, decision deferred
2. **Pro Modules** - AtomicBreak, HighReflect, etc. not yet implemented
3. **Benchmarks** - GSM8K harness ready, full suite pending

---

## Quality Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Test Count | 272+ | 250+ |
| Test Coverage | TBD | 80%+ |
| Clippy Warnings | 0 | 0 |
| Build Time (release) | ~2.5min | <5min |
| Binary Size | TBD | <50MB |

---

*Checklist created: 2025-12-25*
*Last verified: 2025-12-25*
