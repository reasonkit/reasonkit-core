# ReasonKit Core - Release Process

> Comprehensive guide for releasing new versions of reasonkit-core
> Reference: ORCHESTRATOR.md v3.6.0

**Last Updated:** 2025-12-28
**Maintainer:** ReasonKit Team

---

## Table of Contents

1. [Overview](#overview)
2. [Distribution Channels](#distribution-channels)
3. [Semantic Versioning](#semantic-versioning)
4. [Release Workflow](#release-workflow)
5. [Pre-Release Checklist](#pre-release-checklist)
6. [Creating a Release](#creating-a-release)
7. [Automated Pipeline Details](#automated-pipeline-details)
8. [Manual Release Steps](#manual-release-steps)
9. [Post-Release Tasks](#post-release-tasks)
10. [Rollback Procedures](#rollback-procedures)
11. [Troubleshooting](#troubleshooting)
12. [Secrets Configuration](#secrets-configuration)

---

## Overview

ReasonKit Core uses a fully automated release pipeline triggered by semantic version tags. The pipeline builds binaries for 7 platform/architecture combinations, creates Docker images, and publishes to multiple package registries.

### Release Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RELEASE TRIGGER                                    │
│                    git tag v1.0.0 && git push --tags                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: VALIDATION                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Version Check   │  │ Quality Gates   │  │ Security Audit  │              │
│  │ Cargo.toml==Tag │  │ Build/Lint/Test │  │ cargo audit     │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    ▼               ▼               ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                  CHANGELOG GENERATION                        │            │
│  │                    git-cliff (cliff.toml)                   │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ STAGE 2: BINARIES │   │ STAGE 3: DOCKER   │   │ STAGE 4: RELEASE  │
│                   │   │                   │   │                   │
│ - linux-x86_64    │   │ - linux/amd64     │   │ - GitHub Release  │
│ - linux-x86_64-   │   │ - linux/arm64     │   │ - Release Notes   │
│   musl (static)   │   │                   │   │ - SHA256 checksums│
│ - linux-aarch64   │   │ ghcr.io/reasonkit │   │ - install.sh      │
│ - linux-aarch64-  │   │ /reasonkit-core   │   │                   │
│   musl (static)   │   │                   │   │                   │
│ - macos-x86_64    │   │ Tags:             │   │                   │
│ - macos-aarch64   │   │ - v1.0.0          │   │                   │
│ - windows-x86_64  │   │ - 1.0.0           │   │                   │
│                   │   │ - 1.0             │   │                   │
│ SHA256 per binary │   │ - 1               │   │                   │
│                   │   │ - latest          │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ STAGE 5: CRATES   │   │ STAGE 6: NPM      │   │ STAGE 7: PYPI     │
│                   │   │                   │   │                   │
│ cargo publish     │   │ @reasonkit/cli    │   │ reasonkit         │
│ reasonkit-core    │   │ npm wrapper       │   │ Python bindings   │
│                   │   │                   │   │ via maturin       │
│ https://crates.io │   │ https://npmjs.com │   │ https://pypi.org  │
│ /crates/          │   │ /package/         │   │ /project/         │
│ reasonkit-core    │   │ @reasonkit/cli    │   │ reasonkit         │
└───────────────────┘   └───────────────────┘   └───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 9: SUMMARY                                     │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              Release Summary in GitHub Actions UI                │        │
│  │              - All build statuses                               │        │
│  │              - Installation commands                             │        │
│  │              - Asset list with checksums                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Distribution Channels

ReasonKit Core is distributed through multiple channels to maximize accessibility:

| Channel             | Package                            | Command                                           | Primary Use        |
| ------------------- | ---------------------------------- | ------------------------------------------------- | ------------------ |
| **crates.io**       | `reasonkit-core`                   | `cargo install reasonkit-core`                    | Rust developers    |
| **GitHub Releases** | Binary archives                    | `curl -fsSL https://reasonkit.sh/install \| bash` | Universal install  |
| **Docker**          | `ghcr.io/reasonkit/reasonkit-core` | `docker pull ghcr.io/reasonkit/reasonkit-core`    | Containers         |
| **npm**             | `@reasonkit/cli`                   | `npm install -g @reasonkit/cli`                   | Node.js developers |
| **PyPI**            | `reasonkit`                        | `pip install reasonkit`                           | Python developers  |

### Binary Assets Per Release

| Platform | Architecture    | Filename                            | Notes                               |
| -------- | --------------- | ----------------------------------- | ----------------------------------- |
| Linux    | x86_64          | `rk-core-linux-x86_64.tar.gz`       | glibc, most Linux distros           |
| Linux    | x86_64 (static) | `rk-core-linux-x86_64-musl.tar.gz`  | musl, Alpine, minimal containers    |
| Linux    | ARM64           | `rk-core-linux-aarch64.tar.gz`      | glibc, ARM servers, Raspberry Pi 4+ |
| Linux    | ARM64 (static)  | `rk-core-linux-aarch64-musl.tar.gz` | musl, ARM containers                |
| macOS    | x86_64          | `rk-core-macos-x86_64.tar.gz`       | Intel Macs                          |
| macOS    | ARM64           | `rk-core-macos-aarch64.tar.gz`      | Apple Silicon (M1/M2/M3)            |
| Windows  | x86_64          | `rk-core-windows-x86_64.zip`        | Windows 10/11                       |

Each binary archive includes SHA256 checksums for verification.

---

## Semantic Versioning

ReasonKit follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
  1.0.0         - First stable release
  1.1.0         - New backward-compatible features
  1.1.1         - Bug fixes
  2.0.0         - Breaking changes
  1.2.0-alpha.1 - Pre-release alpha
  1.2.0-beta.2  - Pre-release beta
  1.2.0-rc.1    - Release candidate
```

### Version Increment Guidelines

| Change Type                        | Version Bump  | Examples                                   |
| ---------------------------------- | ------------- | ------------------------------------------ |
| Breaking API changes               | MAJOR         | Remove public function, change return type |
| New features (backward compatible) | MINOR         | Add new ThinkTool, add CLI flag            |
| Bug fixes                          | PATCH         | Fix crash, correct output format           |
| Pre-release                        | Append suffix | `-alpha.1`, `-beta.2`, `-rc.1`             |

### Pre-release Behavior

Pre-release versions (containing `-` suffix):

- Marked as "Pre-release" on GitHub
- NOT published to crates.io
- NOT tagged as `latest` in Docker
- npm/PyPI publish is skipped

---

## Release Workflow

### Quick Release (Most Common)

```bash
# 1. Ensure you're on main with clean working directory
git checkout main
git pull origin main
git status  # Should show no changes

# 2. Update version in Cargo.toml
# Edit Cargo.toml: version = "1.1.0"

# 3. Run quality gates
cargo build --release
cargo clippy -- -D warnings
cargo fmt --check
cargo test --lib

# 4. Commit version bump
git add Cargo.toml Cargo.lock
git commit -m "chore: release v1.1.0"

# 5. Create and push tag
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main
git push origin v1.1.0

# 6. Monitor release at:
# https://github.com/reasonkit/reasonkit-core/actions
```

### Using cargo-release (Recommended for Regular Releases)

Install cargo-release for automated version management:

```bash
cargo install cargo-release

# Patch release (1.0.0 -> 1.0.1)
cargo release patch --execute

# Minor release (1.0.0 -> 1.1.0)
cargo release minor --execute

# Major release (1.0.0 -> 2.0.0)
cargo release major --execute

# Pre-release
cargo release --execute -- 1.1.0-alpha.1
```

Add to `Cargo.toml` for cargo-release configuration:

```toml
[package.metadata.release]
sign-commit = false
sign-tag = false
pre-release-commit-message = "chore: release {{version}}"
post-release-commit-message = "chore: bump to {{next_version}}-dev"
tag-message = "Release {{version}}"
tag-name = "v{{version}}"
tag-prefix = "v"
pre-release-replacements = [
    { file = "README.md", search = "reasonkit-core = \"[^\"]+\"", replace = "reasonkit-core = \"{{version}}\"" }
]
```

---

## Pre-Release Checklist

Before creating a release, verify all items:

### Code Quality (CONS-009)

- [ ] **Gate 1: Build** - `cargo build --release` passes
- [ ] **Gate 2: Lint** - `cargo clippy -- -D warnings` reports 0 errors
- [ ] **Gate 3: Format** - `cargo fmt --check` passes
- [ ] **Gate 4: Tests** - `cargo test --lib` passes (100%)
- [ ] **Gate 5: Bench** - `cargo bench --no-run` compiles

### Documentation

- [ ] CHANGELOG.md is up to date (or git-cliff will generate)
- [ ] README.md version references are correct
- [ ] API documentation builds: `cargo doc --no-deps`

### Security

- [ ] `cargo audit` shows no critical vulnerabilities
- [ ] No hardcoded secrets in codebase
- [ ] Dependencies are from trusted sources

### Version

- [ ] Cargo.toml version matches planned release
- [ ] Version follows semantic versioning
- [ ] No duplicate version tags exist

### Quick Check Script

Run this before releasing:

```bash
#!/bin/bash
# scripts/pre-release-check.sh

set -e

echo "=== Pre-Release Verification ==="

echo "1. Checking version..."
VERSION=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "reasonkit-core") | .version')
echo "   Version: $VERSION"

echo "2. Running quality gates..."
cargo build --release --locked
cargo clippy -- -D warnings
cargo fmt --check
cargo test --lib --locked

echo "3. Security audit..."
cargo audit || echo "   Warning: audit issues found (non-blocking)"

echo "4. Documentation build..."
cargo doc --no-deps

echo "5. Package verification..."
cargo package --list | head -20
SIZE=$(du -h target/package/reasonkit-core-*.crate | cut -f1)
echo "   Package size: $SIZE"

echo ""
echo "=== All checks passed! Ready to release v$VERSION ==="
```

---

## Creating a Release

### Method 1: Git Tag (Automated)

The standard way to create a release:

```bash
# Create annotated tag
git tag -a v1.1.0 -m "Release v1.1.0

Features:
- Added new ThinkTool: AtomicBreak
- Improved RAG pipeline performance

Bug Fixes:
- Fixed memory leak in embedding cache
- Corrected CLI output formatting
"

# Push tag to trigger release
git push origin v1.1.0
```

### Method 2: GitHub UI

1. Go to https://github.com/reasonkit/reasonkit-core/releases
2. Click "Draft a new release"
3. Create a new tag (e.g., `v1.1.0`)
4. Fill in release notes
5. Publish release

### Method 3: GitHub CLI

```bash
gh release create v1.1.0 \
  --title "ReasonKit Core v1.1.0" \
  --notes-file RELEASE_NOTES.md \
  --prerelease  # Optional: for pre-releases
```

### Method 4: Workflow Dispatch (Manual Trigger)

For testing or special cases:

1. Go to Actions > Release workflow
2. Click "Run workflow"
3. Select options:
   - `dry_run`: Test without publishing
   - `skip_docker`: Skip Docker build
   - `skip_npm`: Skip npm publish
   - `skip_pypi`: Skip PyPI publish

---

## Automated Pipeline Details

### Stage 1: Validation

**Purpose:** Verify release readiness

- Extracts version from tag
- Validates Cargo.toml version matches tag
- Runs all 5 quality gates (CONS-009)
- Performs security audit
- Generates changelog via git-cliff

**Failure:** Blocks entire release

### Stage 2: Binary Builds

**Purpose:** Cross-compile for all platforms

Matrix builds run in parallel:

| Target                     | OS               | Runner         | Cross?      |
| -------------------------- | ---------------- | -------------- | ----------- |
| x86_64-unknown-linux-gnu   | Linux glibc      | ubuntu-22.04   | No          |
| x86_64-unknown-linux-musl  | Linux musl       | ubuntu-22.04   | No          |
| aarch64-unknown-linux-gnu  | Linux ARM64      | ubuntu-22.04   | Yes (cross) |
| aarch64-unknown-linux-musl | Linux ARM64 musl | ubuntu-22.04   | Yes (cross) |
| x86_64-apple-darwin        | macOS Intel      | macos-13       | No          |
| aarch64-apple-darwin       | macOS ARM        | macos-14       | No          |
| x86_64-pc-windows-msvc     | Windows          | windows-latest | No          |

Each binary is:

1. Built with `--release --locked`
2. Stripped of debug symbols
3. Packaged as `.tar.gz` (Unix) or `.zip` (Windows)
4. SHA256 checksum generated

### Stage 3: Docker Build

**Purpose:** Create multi-arch container images

- Platforms: linux/amd64, linux/arm64
- Registry: ghcr.io/reasonkit/reasonkit-core
- Tags: version, major.minor, major, latest (stable only)
- Includes SBOM and provenance attestation

### Stage 4: GitHub Release

**Purpose:** Create release with all assets

Uploads:

- All binary archives
- All SHA256 checksums
- Combined SHA256SUMS.txt
- install.sh script
- RELEASE_NOTES.md

### Stage 5: crates.io Publish

**Purpose:** Publish Rust crate

- Only for stable releases (no pre-releases)
- Requires `CARGO_REGISTRY_TOKEN` secret
- Validates package before publish

### Stage 6: npm Publish

**Purpose:** Publish JavaScript wrapper

- Creates npm package with post-install script
- Downloads correct binary for platform
- Requires `NPM_TOKEN` secret

### Stage 7: PyPI Publish

**Purpose:** Publish Python bindings

- Uses maturin for wheel building
- Requires `PYPI_API_TOKEN` secret

---

## Manual Release Steps

If the automated pipeline fails or for special releases:

### Manual Binary Build

```bash
# Linux x86_64
cargo build --release --target x86_64-unknown-linux-gnu
strip target/x86_64-unknown-linux-gnu/release/rk-core
cd target/x86_64-unknown-linux-gnu/release
tar czf rk-core-linux-x86_64.tar.gz rk-core
sha256sum rk-core-linux-x86_64.tar.gz > rk-core-linux-x86_64.tar.gz.sha256
```

### Manual crates.io Publish

```bash
# Login (one-time)
cargo login

# Publish
cargo publish --locked
```

### Manual Docker Build

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/reasonkit/reasonkit-core:1.1.0 \
  --tag ghcr.io/reasonkit/reasonkit-core:latest \
  --push \
  .
```

---

## Post-Release Tasks

After a successful release:

### Verification

```bash
# Verify crates.io
cargo search reasonkit-core

# Verify installation
cargo install reasonkit-core
rk-core --version

# Verify Docker
docker run ghcr.io/reasonkit/reasonkit-core:latest --version

# Verify install script
curl -fsSL https://github.com/reasonkit/reasonkit-core/releases/latest/download/install.sh | bash
```

### Announcements

- [ ] Update website version badge
- [ ] Post release announcement (if major)
- [ ] Update documentation site
- [ ] Notify community channels

### Next Version Prep

```bash
# Start development on next version
git checkout -b develop-v1.2.0

# Bump Cargo.toml to next dev version
# version = "1.2.0-dev"

git add Cargo.toml
git commit -m "chore: begin v1.2.0 development"
```

---

## Rollback Procedures

### If Release is Broken

```bash
# 1. Yank crate from crates.io (cannot be undone!)
cargo yank --version 1.1.0

# 2. Delete GitHub release
gh release delete v1.1.0 --yes

# 3. Delete tag
git push --delete origin v1.1.0
git tag -d v1.1.0

# 4. Create fixed release with patch version
# (e.g., v1.1.1 with fix)
```

### Docker Rollback

```bash
# Tag previous version as latest
docker pull ghcr.io/reasonkit/reasonkit-core:1.0.0
docker tag ghcr.io/reasonkit/reasonkit-core:1.0.0 ghcr.io/reasonkit/reasonkit-core:latest
docker push ghcr.io/reasonkit/reasonkit-core:latest
```

---

## Troubleshooting

### Common Issues

#### Version Mismatch Error

```
Error: Version mismatch! Tag=v1.1.0, Cargo.toml=v1.0.0
```

**Solution:** Update Cargo.toml version before tagging:

```bash
# Edit Cargo.toml
git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to 1.1.0"
git tag -a v1.1.0 -m "Release v1.1.0"
```

#### Cross-Compilation Failure

```
Error: failed to run custom build command for `ring`
```

**Solution:** The `cross` tool handles most cases. If failing:

```bash
# Try building without cross
cargo build --release --target aarch64-unknown-linux-gnu

# Or use Docker-based cross
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu
```

#### crates.io Publish Failure

```
Error: the remote server responded with an error: crate version already exists
```

**Solution:** Version already published. Increment patch version.

#### Package Too Large

```
Error: package exceeds maximum allowed size
```

**Solution:** Check `exclude` patterns in Cargo.toml:

```toml
exclude = [
    "data/**",
    "benchmarks/**",
    "*.md",
]
```

### Debug Workflow

Run with dry_run to test without publishing:

1. Go to Actions > Release
2. Run workflow
3. Set `dry_run: true`
4. Review logs for any issues

---

## Secrets Configuration

Required GitHub repository secrets:

| Secret                 | Purpose                     | How to Obtain                           |
| ---------------------- | --------------------------- | --------------------------------------- |
| `CARGO_REGISTRY_TOKEN` | Publish to crates.io        | https://crates.io/settings/tokens       |
| `NPM_TOKEN`            | Publish to npm              | https://www.npmjs.com/settings/~/tokens |
| `PYPI_API_TOKEN`       | Publish to PyPI             | https://pypi.org/manage/account/token/  |
| `DOCKERHUB_USERNAME`   | Docker Hub login (optional) | Docker Hub account                      |
| `DOCKERHUB_TOKEN`      | Docker Hub auth (optional)  | Docker Hub access token                 |

### Setting Secrets

1. Go to repository Settings
2. Security > Secrets and variables > Actions
3. Click "New repository secret"
4. Add each secret

### Minimum Required

For a basic release, only `GITHUB_TOKEN` (automatic) is required. Package registry publishing will skip if tokens are not set.

---

## Changelog Configuration

The release pipeline uses [git-cliff](https://git-cliff.org/) for changelog generation.

Configuration file: `cliff.toml`

Commit message format (Conventional Commits):

- `feat:` - Features
- `fix:` - Bug Fixes
- `perf:` - Performance
- `refactor:` - Refactoring
- `docs:` - Documentation
- `test:` - Testing
- `chore:` - Miscellaneous
- `build:` - Build System
- `ci:` - CI/CD

Example commit messages:

```bash
git commit -m "feat: add AtomicBreak ThinkTool"
git commit -m "fix: resolve memory leak in embedding cache"
git commit -m "perf: optimize BM25 scoring by 40%"
git commit -m "docs: update CLI reference"
```

---

## Summary

The ReasonKit Core release process is:

1. **Prepare:** Update version in Cargo.toml, run quality gates
2. **Tag:** Create semantic version tag (`v1.0.0`)
3. **Push:** Push tag to trigger automated pipeline
4. **Monitor:** Watch Actions for build progress
5. **Verify:** Test installation from all channels
6. **Announce:** Notify users of new release

For issues, check the Actions logs or create an issue at:
https://github.com/reasonkit/reasonkit-core/issues

---

_ReasonKit Core Release Process v1.0.0 | Apache 2.0_
*https://reasonkit.sh*
