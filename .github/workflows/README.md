# ReasonKit Core - CI/CD Pipeline Documentation

> Comprehensive GitHub Actions workflows for quality enforcement, security, and release automation

## Overview

The reasonkit-core project uses **5 primary GitHub Actions workflows** that enforce CONS-009 quality gates and automate the entire build-test-release lifecycle.

```
Workflows:
├── quality-gates.yml    ✅ MANDATORY: 5 quality gates + security
├── release.yml          📦 Multi-platform builds + Docker + crates.io
├── ci.yml               🔄 Comprehensive CI (cross-platform, coverage)
├── security.yml         🔒 Security scanning (daily + PR)
└── benchmark.yml        ⚡ Performance monitoring
```

---

## 1. Quality Gates Workflow

**File:** `quality-gates.yml`
**Trigger:** Every push/PR to main, develop, feature/\*
**Purpose:** Enforce CONS-009 mandatory quality gates

### Gates

| Gate       | Check      | Blocking | Command                              |
| ---------- | ---------- | -------- | ------------------------------------ |
| **Gate 1** | Build      | ✅ YES   | `cargo build --release --locked`     |
| **Gate 2** | Lint       | ✅ YES   | `cargo clippy -- -D warnings`        |
| **Gate 3** | Format     | ✅ YES   | `cargo fmt --check`                  |
| **Gate 4** | Tests      | ✅ YES   | `cargo test --all-features --locked` |
| **Gate 5** | Benchmarks | ⚠️ SOFT  | `cargo bench` (conditional)          |

### Key Features

- **Fast caching** with `Swatinem/rust-cache@v2`
- **Concurrency control** (auto-cancel in-progress runs)
- **Comprehensive metrics** (LOC, TODOs, unsafe blocks)
- **Security audit** with cargo-audit and cargo-deny
- **Final status check** (fails if any gate fails)

### Local Testing

```bash
# Run all gates locally (before pushing)
cd reasonkit-core

# Gate 1: Build
cargo build --release --locked

# Gate 2: Lint
cargo clippy --all-features -- -D warnings

# Gate 3: Format
cargo fmt --check

# Gate 4: Tests
cargo test --all-features --locked

# Gate 5: Benchmarks
cargo bench --bench retrieval_bench
```

---

## 2. Release Workflow

**File:** `release.yml`
**Trigger:** Git tags `v*.*.*` or manual workflow_dispatch
**Purpose:** Automated multi-platform releases

### Build Matrix

| Platform             | Target                      | Binary Name                        |
| -------------------- | --------------------------- | ---------------------------------- |
| Linux x86_64 (glibc) | `x86_64-unknown-linux-gnu`  | `rk-core-linux-x86_64.tar.gz`      |
| Linux x86_64 (musl)  | `x86_64-unknown-linux-musl` | `rk-core-linux-x86_64-musl.tar.gz` |
| Linux ARM64          | `aarch64-unknown-linux-gnu` | `rk-core-linux-aarch64.tar.gz`     |
| macOS x86_64         | `x86_64-apple-darwin`       | `rk-core-macos-x86_64.tar.gz`      |
| macOS ARM64          | `aarch64-apple-darwin`      | `rk-core-macos-aarch64.tar.gz`     |
| Windows x86_64       | `x86_64-pc-windows-msvc`    | `rk-core-windows-x86_64.exe.zip`   |

### Release Process

1. **Validate** - Run quality gates + version tag verification
2. **Build Artifacts** - Cross-compile for all platforms
3. **Docker Images** - Build multi-arch images (amd64, arm64)
4. **Changelog** - Auto-generate with git-cliff
5. **GitHub Release** - Create release with all binaries
6. **crates.io** - Publish to Rust package registry
7. **Install Script** - Generate one-liner installer

### Creating a Release

```bash
# 1. Update version in Cargo.toml
vim Cargo.toml  # version = "0.2.0"

# 2. Commit and tag
git add Cargo.toml
git commit -m "chore: bump version to 0.2.0"
git tag v0.2.0
git push origin main --tags

# 3. GitHub Actions automatically:
#    - Builds all platforms
#    - Creates GitHub release
#    - Publishes to crates.io (if CARGO_REGISTRY_TOKEN set)
#    - Builds Docker images
```

### Docker Images

```bash
# Pull latest release
docker pull ghcr.io/reasonkit/reasonkit-core:latest

# Run with specific version
docker run ghcr.io/reasonkit/reasonkit-core:v0.1.0 --help

# Run with mounted data
docker run -v $(pwd)/data:/app/data ghcr.io/reasonkit/reasonkit-core:latest query "search term"
```

### Manual Workflow Dispatch

You can trigger releases manually from GitHub Actions UI with options:

- **dry_run** - Test the release process without publishing
- **skip_docker** - Skip Docker image build

---

## 3. CI Workflow

**File:** `ci.yml`
**Trigger:** Push/PR to main, develop
**Purpose:** Comprehensive cross-platform testing

### Features

- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Rust version matrix** (stable, beta, nightly, MSRV)
- **Code coverage** with cargo-tarpaulin
- **Cross-compilation** verification
- **Dependency tree** analysis

### Coverage Reports

Coverage is automatically uploaded to Codecov on push events.

---

## 4. Security Workflow

**File:** `security.yml`
**Trigger:** Daily schedule (00:00 UTC) + PR
**Purpose:** Comprehensive security scanning

### Security Checks

| Check            | Tool             | Purpose                     |
| ---------------- | ---------------- | --------------------------- |
| Dependency Audit | cargo-audit      | CVE scanning                |
| Supply Chain     | cargo-deny       | License/source verification |
| License Check    | cargo-license    | GPL detection               |
| Secret Scanning  | Gitleaks         | Hardcoded secrets           |
| SAST             | Clippy + Semgrep | Code vulnerabilities        |
| SBOM             | cargo-sbom       | CycloneDX + SPDX            |

### SBOM Generation

Software Bill of Materials (SBOM) is generated for every release:

- **CycloneDX JSON** (industry standard)
- **SPDX JSON** (Linux Foundation standard)

---

## 5. Benchmark Workflow

**File:** `benchmark.yml`
**Trigger:** Push to main + performance label
**Purpose:** Performance regression detection

### Metrics Tracked

- **Criterion benchmarks** (retrieval, indexing)
- **Binary size** (release builds)
- **Compile time** (clean + incremental)
- **Dependency analysis** (duplicate detection)
- **Flamegraphs** (CPU profiling)

### Performance Alerts

Automatic PR comments if:

- Benchmark regression > 10%
- Binary size increase > 5%
- New duplicate dependencies

---

## Required Secrets

Configure these in GitHub Settings → Secrets:

| Secret                 | Required | Purpose                  |
| ---------------------- | -------- | ------------------------ |
| `CARGO_REGISTRY_TOKEN` | Optional | crates.io publishing     |
| `GITLEAKS_LICENSE`     | Optional | Enhanced secret scanning |

**Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions.

---

## Branch Protection Rules

Recommended settings for `main` branch:

```yaml
Required status checks:
  - Gate 1: Build
  - Gate 2: Clippy
  - Gate 3: Format
  - Gate 4: Tests
  - CI Success

Require branches to be up to date: ✅
Require signed commits: ✅
Require pull request reviews: 1
Dismiss stale reviews: ✅
```

---

## Workflow Badges

Add these to `README.md`:

```markdown
[![Quality Gates](https://github.com/reasonkit/reasonkit-core/actions/workflows/quality-gates.yml/badge.svg)](https://github.com/reasonkit/reasonkit-core/actions/workflows/quality-gates.yml)
[![Security](https://github.com/reasonkit/reasonkit-core/actions/workflows/security.yml/badge.svg)](https://github.com/reasonkit/reasonkit-core/actions/workflows/security.yml)
[![Release](https://github.com/reasonkit/reasonkit-core/actions/workflows/release.yml/badge.svg)](https://github.com/reasonkit/reasonkit-core/actions/workflows/release.yml)
```

---

## Performance Optimization

### Caching Strategy

All workflows use `Swatinem/rust-cache@v2` for optimal caching:

- **Cargo registry** - Downloaded crates
- **Git dependencies** - Git-based deps
- **Build artifacts** - Incremental compilation

### Concurrency Control

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

This automatically cancels in-progress runs when new commits are pushed.

---

## Troubleshooting

### Build Failures

**Problem:** Gate 1 (Build) fails with "Cargo.lock out of date"

```bash
# Solution: Update Cargo.lock
cargo update --workspace
git add Cargo.lock
git commit -m "chore: update Cargo.lock"
```

### Clippy Failures

**Problem:** Gate 2 (Lint) fails with new warnings

```bash
# Solution: Fix clippy warnings
cargo clippy --all-features --fix
git add .
git commit -m "fix: clippy warnings"
```

### Format Failures

**Problem:** Gate 3 (Format) fails

```bash
# Solution: Auto-format code
cargo fmt
git add .
git commit -m "style: cargo fmt"
```

### Test Failures

**Problem:** Gate 4 (Tests) fails

```bash
# Solution: Run tests locally first
cargo test --all-features

# Debug specific test
cargo test test_name -- --nocapture
```

### Release Tag Mismatch

**Problem:** Release workflow fails with "Version mismatch"

```bash
# Solution: Ensure Cargo.toml version matches tag
# Tag v0.2.0 requires version = "0.2.0" in Cargo.toml
```

---

## Maintenance

### Weekly Tasks

- Review security audit reports
- Check for dependency updates
- Review benchmark trends

### Monthly Tasks

- Update GitHub Actions versions
- Review and update SBOM
- Audit workflow performance

### Quarterly Tasks

- Review and update branch protection rules
- Audit secret rotation
- Performance baseline reset

---

## References

- **ORCHESTRATOR.md** - Master orchestration document
- **QA_PLAN.md** - Quality assurance plan
- **REVIEW_PROTOCOL.md** - Code review protocol
- **GitHub Actions Docs** - https://docs.github.com/actions

---

**Version:** 1.0.0
**Last Updated:** 2025-12-23
**Maintained by:** ReasonKit DevOps Team
