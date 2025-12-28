# ReasonKit Deployment Implementation Plan

**Created:** 2025-12-25
**Status:** Ready for Implementation
**Based on:** [RUST_CLI_DEPLOYMENT_RESEARCH.md](./RUST_CLI_DEPLOYMENT_RESEARCH.md)

---

## Overview

This document provides a 4-week implementation roadmap for deploying ReasonKit as a production-ready Rust CLI tool, following industry best practices from ripgrep, bat, fd, and other leading tools.

---

## Phase 1: Foundation (Week 1)

**Goal:** Establish core deployment infrastructure

### Day 1-2: Optimize Cargo.toml

**Task:** Configure release profile for optimal binaries

```toml
# reasonkit-core/Cargo.toml

[profile.release]
strip = true          # Remove debug symbols
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
opt-level = "z"      # Optimize for size
panic = "abort"      # Smaller binary

[package.metadata.binstall]
pkg-url = "{ repo }/releases/download/v{ version }/{ name }-{ target }{ archive-suffix }"
bin-dir = "{ bin }{ binary-ext }"
pkg-fmt = "tgz"
```

**Verification:**
```bash
cargo build --release
ls -lh target/release/rk-core  # Check binary size
./target/release/rk-core --version  # Test functionality
```

**Expected Result:** Binary size < 5 MB (optimized from debug build)

---

### Day 2-3: GitHub Actions CI/CD

**Task:** Set up automated testing and release workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --check
```

**File:** `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - target: x86_64-unknown-linux-musl
            os: ubuntu-latest
            build_cmd: |
              sudo apt-get update
              sudo apt-get install -y musl-tools
              rustup target add x86_64-unknown-linux-musl
              cargo build --release --target x86_64-unknown-linux-musl
          - target: x86_64-apple-darwin
            os: macos-13  # Intel
            build_cmd: cargo build --release --target x86_64-apple-darwin
          - target: aarch64-apple-darwin
            os: macos-14  # Apple Silicon
            build_cmd: cargo build --release --target aarch64-apple-darwin
          - target: x86_64-pc-windows-msvc
            os: windows-latest
            build_cmd: cargo build --release --target x86_64-pc-windows-msvc

    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - uses: Swatinem/rust-cache@v2

      - name: Build
        run: ${{ matrix.build_cmd }}

      - name: Strip binary (Unix)
        if: matrix.os != 'windows-latest'
        run: |
          strip target/${{ matrix.target }}/release/rk-core || true

      - name: Create archive
        id: archive
        run: |
          cd target/${{ matrix.target }}/release
          if [ "${{ matrix.os }}" = "windows-latest" ]; then
            7z a ../../../rk-core-${{ matrix.target }}.zip rk-core.exe
            echo "archive=rk-core-${{ matrix.target }}.zip" >> $GITHUB_OUTPUT
          else
            tar czf ../../../rk-core-${{ matrix.target }}.tar.gz rk-core
            echo "archive=rk-core-${{ matrix.target }}.tar.gz" >> $GITHUB_OUTPUT
          fi

      - name: Generate checksum
        run: |
          if [ "${{ matrix.os }}" = "windows-latest" ]; then
            certutil -hashfile ${{ steps.archive.outputs.archive }} SHA256 > ${{ steps.archive.outputs.archive }}.sha256
          else
            shasum -a 256 ${{ steps.archive.outputs.archive }} > ${{ steps.archive.outputs.archive }}.sha256
          fi

      - name: Upload to GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            ${{ steps.archive.outputs.archive }}
            ${{ steps.archive.outputs.archive }}.sha256
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Verification:**
```bash
# Create test tag
git tag v0.1.0-test
git push origin v0.1.0-test

# Watch GitHub Actions run
# Check that binaries are created and uploaded
```

---

### Day 4-5: Shell Completions

**Task:** Generate shell completions for all major shells

**File:** `reasonkit-core/build.rs`

```rust
use clap::CommandFactory;
use clap_complete::{generate_to, shells::*};
use std::env;
use std::io::Error;

include!("src/cli.rs");  // Or wherever your CLI struct is

fn main() -> Result<(), Error> {
    let outdir = match env::var_os("OUT_DIR") {
        None => return Ok(()),
        Some(outdir) => outdir,
    };

    let mut cmd = Cli::command();
    let bin_name = "rk-core";

    generate_to(Bash, &mut cmd, bin_name, &outdir)?;
    generate_to(Zsh, &mut cmd, bin_name, &outdir)?;
    generate_to(Fish, &mut cmd, bin_name, &outdir)?;
    generate_to(PowerShell, &mut cmd, bin_name, &outdir)?;

    println!("cargo:rerun-if-changed=src/cli.rs");
    Ok(())
}
```

**Update Cargo.toml:**

```toml
[build-dependencies]
clap = { version = "4", features = ["derive"] }
clap_complete = "4"
```

**Alternative: Runtime generation binary:**

```rust
// src/bin/generate-completions.rs
use clap::CommandFactory;
use clap_complete::{generate, shells::*};
use std::io;

fn main() {
    let mut cmd = reasonkit_core::cli::Cli::command();

    println!("Generating shell completions...");

    let out_dir = std::path::PathBuf::from("completions");
    std::fs::create_dir_all(&out_dir).unwrap();

    for shell in [Bash, Zsh, Fish, PowerShell] {
        let mut file = std::fs::File::create(
            out_dir.join(format!("rk-core.{}", shell.file_name("rk-core")))
        ).unwrap();
        generate(shell, &mut cmd, "rk-core", &mut file);
    }

    println!("Completions generated in completions/");
}
```

**Verification:**
```bash
cargo run --bin generate-completions
ls -la completions/
source completions/rk-core.bash  # Test Bash completions
```

---

### Day 6-7: Installation Script

**Task:** Create universal installation script

**File:** `reasonkit-site/install.sh`

```bash
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO="your-username/reasonkit-core"
BINARY_NAME="rk-core"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

echo -e "${GREEN}ReasonKit Installer${NC}"
echo "================================"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    x86_64|amd64) ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
esac

case "$OS" in
    linux)
        TARGET="x86_64-unknown-linux-musl"
        ARCHIVE_EXT="tar.gz"
        ;;
    darwin)
        if [[ "$ARCH" == "aarch64" ]]; then
            TARGET="aarch64-apple-darwin"
        else
            TARGET="x86_64-apple-darwin"
        fi
        ARCHIVE_EXT="tar.gz"
        ;;
    *)
        echo -e "${RED}Unsupported OS: $OS${NC}"
        exit 1
        ;;
esac

echo "Detected: $OS $ARCH (target: $TARGET)"

# Get latest release
echo "Fetching latest version..."
LATEST_RELEASE=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_RELEASE" ]; then
    echo -e "${RED}Failed to fetch latest release${NC}"
    exit 1
fi

VERSION=${LATEST_RELEASE#v}
echo "Latest version: $VERSION"

# Download URL
ARCHIVE_NAME="${BINARY_NAME}-${TARGET}.${ARCHIVE_EXT}"
URL="https://github.com/$REPO/releases/download/$LATEST_RELEASE/$ARCHIVE_NAME"
CHECKSUM_URL="${URL}.sha256"

echo "Downloading from: $URL"

# Download binary
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

cd "$TMP_DIR"
if ! curl -fsSL "$URL" -o "$ARCHIVE_NAME"; then
    echo -e "${RED}Failed to download $ARCHIVE_NAME${NC}"
    exit 1
fi

# Verify checksum
if curl -fsSL "$CHECKSUM_URL" -o checksum.txt; then
    echo "Verifying checksum..."
    if command -v shasum >/dev/null; then
        shasum -a 256 -c checksum.txt || { echo -e "${RED}Checksum verification failed${NC}"; exit 1; }
    elif command -v sha256sum >/dev/null; then
        sha256sum -c checksum.txt || { echo -e "${RED}Checksum verification failed${NC}"; exit 1; }
    else
        echo -e "${YELLOW}Warning: Cannot verify checksum (shasum not found)${NC}"
    fi
fi

# Extract
echo "Extracting..."
tar xzf "$ARCHIVE_NAME"

# Install
mkdir -p "$INSTALL_DIR"
mv "$BINARY_NAME" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo -e "${GREEN}Successfully installed $BINARY_NAME to $INSTALL_DIR${NC}"

# Check if in PATH
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    echo -e "${YELLOW}Warning: $INSTALL_DIR is not in your PATH${NC}"
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
fi

echo ""
echo "Run '$BINARY_NAME --help' to get started!"
```

**Make executable and test:**
```bash
chmod +x reasonkit-site/install.sh
./reasonkit-site/install.sh
```

**Host on website:**
```bash
# Upload to reasonkit.sh/install.sh
# Users can install with:
# curl -fsSL https://reasonkit.sh/install.sh | bash
```

---

## Phase 2: Distribution (Week 2)

### Day 8-9: cargo-dist Setup

**Task:** Automate release artifact generation

```bash
# Install cargo-dist
cargo install cargo-dist

# Initialize
cd reasonkit-core
cargo dist init

# This creates dist configuration in Cargo.toml
```

**Review and commit:**
```toml
# Auto-generated in Cargo.toml
[workspace.metadata.dist]
cargo-dist-version = "0.30.2"
ci = ["github"]
installers = ["shell", "homebrew"]
targets = [
    "x86_64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc"
]
```

**Test locally:**
```bash
cargo dist build
cargo dist plan
```

---

### Day 10-11: Homebrew Formula

**Task:** Create Homebrew tap for macOS distribution

**Step 1: Create tap repository**
```bash
# On GitHub, create: homebrew-tap
# Repository URL: https://github.com/username/homebrew-tap
```

**Step 2: Create formula**

```bash
mkdir -p Formula
cd Formula
```

**File:** `Formula/reasonkit.rb`

```ruby
class Reasonkit < Formula
  desc "Structured reasoning protocols for LLMs"
  homepage "https://reasonkit.sh"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/username/reasonkit-core/releases/download/v0.1.0/rk-core-aarch64-apple-darwin.tar.gz"
      sha256 "CHECKSUM_HERE"
    else
      url "https://github.com/username/reasonkit-core/releases/download/v0.1.0/rk-core-x86_64-apple-darwin.tar.gz"
      sha256 "CHECKSUM_HERE"
    end
  end

  on_linux do
    url "https://github.com/username/reasonkit-core/releases/download/v0.1.0/rk-core-x86_64-unknown-linux-musl.tar.gz"
    sha256 "CHECKSUM_HERE"
  end

  def install
    bin.install "rk-core"

    # Install shell completions if they exist
    if (buildpath/"completions").exist?
      bash_completion.install "completions/rk-core.bash" => "rk-core"
      zsh_completion.install "completions/_rk-core"
      fish_completion.install "completions/rk-core.fish"
    end
  end

  test do
    assert_match "rk-core", shell_output("#{bin}/rk-core --version")
  end
end
```

**Step 3: Test locally**
```bash
# Install from local tap
brew tap username/tap
brew install --build-from-source Formula/reasonkit.rb

# Test
rk-core --version
```

**Step 4: Update after each release**
```bash
# Get new SHA256
shasum -a 256 rk-core-x86_64-apple-darwin.tar.gz

# Update formula with new version and checksum
# Commit and push
```

---

### Day 12-13: cargo-binstall Metadata

**Task:** Enable fast binary installation

**Update Cargo.toml:**

```toml
[package.metadata.binstall]
pkg-url = "{ repo }/releases/download/v{ version }/{ name }-{ target }{ archive-suffix }"
bin-dir = "{ bin }{ binary-ext }"
pkg-fmt = "tgz"

[package.metadata.binstall.overrides.x86_64-pc-windows-msvc]
pkg-fmt = "zip"
```

**Test:**
```bash
# After creating a release
cargo binstall reasonkit --dry-run
cargo binstall reasonkit
```

---

### Day 14: Static Linux Binaries

**Task:** Ensure portable Linux binaries with musl

**Update CI workflow:**

```yaml
- name: Install musl tools (Linux)
  if: matrix.os == 'ubuntu-latest'
  run: |
    sudo apt-get update
    sudo apt-get install -y musl-tools

- name: Add musl target
  if: matrix.os == 'ubuntu-latest'
  run: rustup target add x86_64-unknown-linux-musl

- name: Build (Linux)
  if: matrix.os == 'ubuntu-latest'
  run: cargo build --release --target x86_64-unknown-linux-musl
```

**Verify static linking:**
```bash
ldd target/x86_64-unknown-linux-musl/release/rk-core
# Should output: "not a dynamic executable"
```

---

## Phase 3: Automation (Week 3)

### Day 15-16: git-cliff Integration

**Task:** Automated changelog generation

```bash
# Install
cargo install git-cliff

# Initialize config
git cliff --init
```

**File:** `reasonkit-core/cliff.toml`

```toml
[changelog]
header = """
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
body = """
{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | striptags | trim | upper_first }}
    {% for commit in commits %}
        - {% if commit.scope %}**{{commit.scope}}:** {% endif %}{{ commit.message | upper_first }} ([{{ commit.id | truncate(length=7, end="") }}]({{ commit.id }}))
    {% endfor %}
{% endfor %}
"""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
split_commits = false
commit_parsers = [
    { message = "^feat", group = "Features" },
    { message = "^fix", group = "Bug Fixes" },
    { message = "^doc", group = "Documentation" },
    { message = "^perf", group = "Performance" },
    { message = "^refactor", group = "Refactor" },
    { message = "^style", group = "Styling" },
    { message = "^test", group = "Testing" },
    { message = "^chore\\(release\\): prepare for", skip = true },
    { message = "^chore", group = "Miscellaneous Tasks" },
    { body = ".*security", group = "Security" },
]
```

**Update release workflow:**
```yaml
- name: Generate changelog
  run: |
    cargo install git-cliff
    git cliff --latest --output CHANGELOG.md
```

**Manual usage:**
```bash
# Generate for latest version
git cliff --latest -o CHANGELOG.md

# Generate for specific range
git cliff v0.1.0..v0.2.0 -o CHANGELOG.md

# Preview unreleased changes
git cliff --unreleased
```

---

### Day 17-18: cargo-deny Setup

**Task:** Supply chain security and license compliance

```bash
# Install
cargo install cargo-deny

# Initialize
cd reasonkit-core
cargo deny init
```

**File:** `reasonkit-core/deny.toml`

```toml
[advisories]
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"
notice = "warn"
ignore = []

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
]
deny = [
    "GPL-3.0",
    "AGPL-3.0",
]
copyleft = "warn"
allow-osi-fsf-free = "both"
default = "deny"
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "warn"
highlight = "all"
deny = []
skip = []
skip-tree = []

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

**Add to CI:**
```yaml
- name: Security & License Check
  run: |
    cargo install cargo-deny
    cargo deny check
```

**Local usage:**
```bash
# Check everything
cargo deny check

# Check specific category
cargo deny check advisories
cargo deny check licenses
cargo deny check bans
```

---

### Day 19-20: Binary Size Optimization

**Task:** Minimize binary size for faster downloads

**Already done in Cargo.toml (Phase 1), but verify:**

```bash
# Baseline
cargo build
ls -lh target/debug/rk-core

# Optimized release
cargo build --release
ls -lh target/release/rk-core

# With musl (static)
cargo build --release --target x86_64-unknown-linux-musl
ls -lh target/x86_64-unknown-linux-musl/release/rk-core
```

**Optional: UPX compression**

```bash
# Install UPX
brew install upx  # macOS
sudo apt install upx-ucl  # Ubuntu

# Compress
upx --best target/release/rk-core

# Check size
ls -lh target/release/rk-core
```

**Document results:**
```
Before optimization:  10.2 MB (debug)
After --release:       2.1 MB
After opt-level="z":   1.8 MB
After strip:           1.7 MB
After UPX:             0.6 MB
```

---

### Day 21: Release Process Documentation

**Task:** Document the complete release process

**File:** `reasonkit-core/docs/RELEASING.md`

```markdown
# Release Process

## Prerequisites

- [ ] All tests passing (`cargo test`)
- [ ] Clippy clean (`cargo clippy -- -D warnings`)
- [ ] Formatted (`cargo fmt --check`)
- [ ] Security checks pass (`cargo deny check`)
- [ ] CHANGELOG.md updated or auto-generated

## Release Steps

### 1. Version Bump

```bash
# Update version in Cargo.toml
# Update version in README.md if applicable
```

### 2. Generate Changelog

```bash
git cliff --tag v0.1.0 -o CHANGELOG.md
git add CHANGELOG.md
git commit -m "chore(release): prepare for v0.1.0"
```

### 3. Create Tag

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

### 4. Monitor Release

- GitHub Actions builds binaries
- GitHub Release is created automatically
- Binaries are uploaded to release

### 5. Publish to crates.io

```bash
cargo publish
```

### 6. Update Homebrew Formula

```bash
cd ../homebrew-tap/Formula
# Update reasonkit.rb with new version and checksums
# Get checksums from GitHub release
git commit -m "Update reasonkit to v0.1.0"
git push
```

### 7. Post-Release

- [ ] Verify installation: `cargo install reasonkit`
- [ ] Verify binstall: `cargo binstall reasonkit`
- [ ] Verify Homebrew: `brew upgrade reasonkit`
- [ ] Announce on social media
- [ ] Update documentation site
```

---

## Phase 4: Expansion (Week 4)

### Day 22-23: Debian Packages

**Task:** Create .deb packages for Debian/Ubuntu

```bash
# Install cargo-deb
cargo install cargo-deb
```

**Update Cargo.toml:**

```toml
[package.metadata.deb]
maintainer = "Your Name <email@example.com>"
copyright = "2025, Your Name <email@example.com>"
license-file = ["LICENSE", "4"]
extended-description = """\
ReasonKit provides structured reasoning protocols for Large Language Models (LLMs). \
It enables developers to apply systematic thinking patterns to AI interactions."""
depends = "$auto"
section = "utility"
priority = "optional"
assets = [
    ["target/release/rk-core", "usr/bin/", "755"],
    ["README.md", "usr/share/doc/reasonkit/README.md", "644"],
    ["LICENSE", "usr/share/doc/reasonkit/LICENSE", "644"],
    ["completions/rk-core.bash", "usr/share/bash-completion/completions/rk-core", "644"],
    ["completions/_rk-core", "usr/share/zsh/vendor-completions/_rk-core", "644"],
    ["completions/rk-core.fish", "usr/share/fish/vendor_completions.d/rk-core.fish", "644"],
]
```

**Build:**
```bash
cargo deb
# Output: target/debian/reasonkit_0.1.0-1_amd64.deb
```

**Test:**
```bash
sudo dpkg -i target/debian/reasonkit_0.1.0-1_amd64.deb
rk-core --version
sudo dpkg -r reasonkit
```

**Add to release:**
```yaml
# In .github/workflows/release.yml
- name: Build .deb package (Linux)
  if: matrix.os == 'ubuntu-latest'
  run: |
    cargo install cargo-deb
    cargo deb
    mv target/debian/*.deb ./

- name: Upload .deb to release
  if: matrix.os == 'ubuntu-latest'
  uses: softprops/action-gh-release@v1
  with:
    files: '*.deb'
```

---

### Day 24-25: Windows Installers

**Task:** Create Chocolatey package

**Create package structure:**
```
chocolatey/
├── reasonkit.nuspec
└── tools/
    ├── LICENSE.txt
    ├── VERIFICATION.txt
    ├── chocolateyinstall.ps1
    └── chocolateyuninstall.ps1
```

**File:** `chocolatey/reasonkit.nuspec`

```xml
<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2015/06/nuspec.xsd">
  <metadata>
    <id>reasonkit</id>
    <version>0.1.0</version>
    <packageSourceUrl>https://github.com/username/reasonkit-core</packageSourceUrl>
    <owners>Your Name</owners>
    <title>ReasonKit</title>
    <authors>Your Name</authors>
    <projectUrl>https://reasonkit.sh</projectUrl>
    <licenseUrl>https://github.com/username/reasonkit-core/blob/main/LICENSE</licenseUrl>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <projectSourceUrl>https://github.com/username/reasonkit-core</projectSourceUrl>
    <docsUrl>https://reasonkit.sh/docs</docsUrl>
    <bugTrackerUrl>https://github.com/username/reasonkit-core/issues</bugTrackerUrl>
    <tags>rust cli reasoning ai llm</tags>
    <summary>Structured reasoning protocols for LLMs</summary>
    <description>
ReasonKit provides structured reasoning protocols for Large Language Models (LLMs).
It enables developers to apply systematic thinking patterns to AI interactions.
    </description>
  </metadata>
  <files>
    <file src="tools\**" target="tools" />
  </files>
</package>
```

**File:** `chocolatey/tools/chocolateyinstall.ps1`

```powershell
$ErrorActionPreference = 'Stop'

$packageName = 'reasonkit'
$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
$url64 = 'https://github.com/username/reasonkit-core/releases/download/v0.1.0/rk-core-x86_64-pc-windows-msvc.zip'
$checksum64 = 'CHECKSUM_HERE'

$packageArgs = @{
  packageName   = $packageName
  unzipLocation = $toolsDir
  url64bit      = $url64
  checksum64    = $checksum64
  checksumType64= 'sha256'
}

Install-ChocolateyZipPackage @packageArgs
```

**Build and test:**
```powershell
# Pack
choco pack

# Test locally
choco install reasonkit -s . -y

# Verify
rk-core --version

# Uninstall
choco uninstall reasonkit -y
```

---

### Day 26-27: Docker Images

**Task:** Create Docker images for containerized deployments

**File:** `reasonkit-core/Dockerfile`

```dockerfile
# Build stage
FROM rust:1.75 AS builder

WORKDIR /app
COPY . .

# Build with musl for static binary
RUN apt-get update && apt-get install -y musl-tools
RUN rustup target add x86_64-unknown-linux-musl
RUN cargo build --release --target x86_64-unknown-linux-musl

# Runtime stage (minimal)
FROM scratch

COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rk-core /rk-core

ENTRYPOINT ["/rk-core"]
```

**Build and test:**
```bash
docker build -t reasonkit:latest .
docker run --rm reasonkit:latest --version
docker run --rm -v $(pwd):/data reasonkit:latest --help
```

**Multi-stage with Alpine:**
```dockerfile
FROM rust:1.75-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /app
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM alpine:latest
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rk-core /usr/local/bin/
ENTRYPOINT ["rk-core"]
```

**Publish to Docker Hub:**
```bash
docker tag reasonkit:latest username/reasonkit:0.1.0
docker tag reasonkit:latest username/reasonkit:latest
docker push username/reasonkit:0.1.0
docker push username/reasonkit:latest
```

---

### Day 28: Documentation and Polish

**Task:** Final documentation updates

**Update README.md:**

```markdown
# ReasonKit

Structured reasoning protocols for LLMs.

## Installation

### Rust Developers
```bash
cargo install reasonkit
# or faster:
cargo binstall reasonkit
```

### macOS
```bash
brew install username/tap/reasonkit
```

### Linux (Debian/Ubuntu)
```bash
wget https://github.com/username/reasonkit-core/releases/download/v0.1.0/reasonkit_0.1.0_amd64.deb
sudo dpkg -i reasonkit_0.1.0_amd64.deb
```

### Windows (Chocolatey)
```powershell
choco install reasonkit
```

### Universal
```bash
curl -fsSL https://reasonkit.sh/install.sh | bash
```

### From Source
```bash
git clone https://github.com/username/reasonkit-core
cd reasonkit-core
cargo install --path .
```

## Shell Completions

```bash
# Bash
rk-core completions bash > ~/.local/share/bash-completion/completions/rk-core

# Zsh
rk-core completions zsh > ~/.zfunc/_rk-core

# Fish
rk-core completions fish > ~/.config/fish/completions/rk-core.fish
```

## Usage

```bash
rk-core --help
```
```

---

## Success Metrics

### Week 1 (Foundation)
- [ ] Binary size < 5 MB
- [ ] CI/CD pipeline functional
- [ ] Shell completions generated
- [ ] Install script works on macOS and Linux

### Week 2 (Distribution)
- [ ] cargo-dist configured
- [ ] Homebrew formula tested
- [ ] cargo-binstall working
- [ ] Static Linux binaries verified

### Week 3 (Automation)
- [ ] Changelog auto-generated
- [ ] Security checks in CI
- [ ] Binary size optimized
- [ ] Release process documented

### Week 4 (Expansion)
- [ ] .deb packages created
- [ ] Chocolatey package tested
- [ ] Docker images published
- [ ] Documentation complete

---

## Post-Implementation

### Monitoring
- GitHub Actions success rate
- Binary download counts
- Issue reports
- Installation method distribution

### Continuous Improvement
- Update dependencies monthly
- Review security advisories weekly
- Optimize based on user feedback
- Expand platform support as needed

---

## Resources

- [Full Research](./RUST_CLI_DEPLOYMENT_RESEARCH.md)
- [Quick Reference](./DEPLOYMENT_QUICK_REFERENCE.md)
- [Rust CLI Book](https://rust-cli.github.io/book/)
- [cargo-dist Documentation](https://opensource.axo.dev/cargo-dist/)

---

**Status:** Ready for Implementation
**Next Action:** Begin Phase 1, Day 1 - Optimize Cargo.toml
