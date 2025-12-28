# Rust CLI Deployment - Quick Reference Guide

**Last Updated:** 2025-12-25
**Full Research:** [RUST_CLI_DEPLOYMENT_RESEARCH.md](./RUST_CLI_DEPLOYMENT_RESEARCH.md)

---

## Essential Cargo.toml Configuration

```toml
[package]
name = "reasonkit"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/username/reasonkit"
description = "Structured reasoning for LLMs"

[[bin]]
name = "rk-core"
path = "src/main.rs"

[profile.release]
strip = true          # Remove debug symbols (Rust 1.59+)
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
opt-level = "z"      # Optimize for size ("s" or "z")
panic = "abort"      # Smaller binary
```

---

## Cross-Compilation Targets

```bash
# Essential targets
rustup target add x86_64-apple-darwin          # macOS Intel
rustup target add aarch64-apple-darwin         # macOS Apple Silicon
rustup target add x86_64-unknown-linux-musl    # Linux (static)
rustup target add x86_64-pc-windows-msvc       # Windows

# Build for target
cargo build --release --target x86_64-unknown-linux-musl

# Or use cross for zero-setup
cargo install cross
cross build --target aarch64-unknown-linux-gnu
```

---

## GitHub Actions Workflow (Minimal)

```yaml
# .github/workflows/release.yml
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
          - target: x86_64-apple-darwin
            os: macos-latest
          - target: aarch64-apple-darwin
            os: macos-latest
          - target: x86_64-pc-windows-msvc
            os: windows-latest
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --release --target ${{ matrix.target }}
      - uses: softprops/action-gh-release@v1
        with:
          files: target/${{ matrix.target }}/release/rk-core*
```

---

## Shell Completions Generation

```rust
// build.rs or separate binary
use clap::CommandFactory;
use clap_complete::{generate_to, shells::*};

fn main() {
    let mut cmd = Cli::command();
    let out_dir = std::path::PathBuf::from("completions/");

    generate_to(Bash, &mut cmd, "rk-core", &out_dir)?;
    generate_to(Zsh, &mut cmd, "rk-core", &out_dir)?;
    generate_to(Fish, &mut cmd, "rk-core", &out_dir)?;
    generate_to(PowerShell, &mut cmd, "rk-core", &out_dir)?;
}
```

```toml
# Add to Cargo.toml
[dependencies]
clap = { version = "4", features = ["derive"] }

[build-dependencies]
clap = { version = "4", features = ["derive"] }
clap_complete = "4"
```

---

## Homebrew Formula (Basic)

```ruby
# Formula/reasonkit.rb in homebrew-tap repo
class Reasonkit < Formula
  desc "Structured reasoning for LLMs"
  homepage "https://reasonkit.sh"
  url "https://github.com/user/reasonkit/releases/download/v0.1.0/rk-core-0.1.0-x86_64-apple-darwin.tar.gz"
  sha256 "abc123..."
  version "0.1.0"
  license "Apache-2.0"

  def install
    bin.install "rk-core"
    bash_completion.install "completions/rk-core.bash"
    zsh_completion.install "completions/_rk-core"
    fish_completion.install "completions/rk-core.fish"
  end

  test do
    system "#{bin}/rk-core", "--version"
  end
end
```

**Installation:**
```bash
brew tap username/tap
brew install reasonkit
```

---

## cargo-binstall Support

```toml
# Cargo.toml
[package.metadata.binstall]
pkg-url = "{ repo }/releases/download/v{ version }/{ name }-{ target }{ archive-suffix }"
bin-dir = "{ bin }{ binary-ext }"
pkg-fmt = "tgz"
```

**Usage:**
```bash
cargo binstall reasonkit  # Downloads binary instead of compiling
```

---

## Changelog Automation

```bash
# Install git-cliff
cargo install git-cliff

# Generate changelog
git cliff -o CHANGELOG.md

# Or use in Cargo.toml
[package.metadata.git-cliff.changelog]
header = "# Changelog\n\n"
body = """
{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | upper_first }}
    {% for commit in commits %}
        - {{ commit.message }}
    {% endfor %}
{% endfor %}
"""
```

---

## Release Automation

```bash
# Install cargo-release
cargo install cargo-release

# Release workflow
cargo release patch --execute  # 0.1.0 -> 0.1.1
cargo release minor --execute  # 0.1.0 -> 0.2.0
cargo release major --execute  # 0.1.0 -> 1.0.0
```

**What it does:**
1. Runs tests
2. Bumps version
3. Updates changelog
4. Commits changes
5. Creates git tag
6. Publishes to crates.io
7. Pushes to GitHub

---

## Supply Chain Security

```bash
# Install cargo-deny
cargo install cargo-deny

# Initialize
cargo deny init

# Check everything
cargo deny check

# CI integration
- uses: EmbarkStudios/cargo-deny-action@v1
```

**deny.toml example:**
```toml
[advisories]
vulnerability = "deny"

[licenses]
unlicensed = "deny"
allow = ["MIT", "Apache-2.0", "BSD-3-Clause"]

[bans]
multiple-versions = "warn"
```

---

## Binary Size Optimization

```bash
# Build optimized
cargo build --release --target x86_64-unknown-linux-musl

# Strip symbols (if not using strip = true)
strip target/x86_64-unknown-linux-musl/release/rk-core

# Optional: UPX compression (30-50% reduction)
upx --best target/x86_64-unknown-linux-musl/release/rk-core
```

**Expected results:**
- Original debug: ~10 MB
- After --release: ~2 MB
- After optimizations: ~800 KB
- After UPX: ~300 KB

---

## Installation Script Template

```bash
#!/bin/bash
# install.sh

set -e

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux) TARGET="x86_64-unknown-linux-musl" ;;
  darwin) TARGET="x86_64-apple-darwin" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

VERSION="0.1.0"
URL="https://github.com/user/reasonkit/releases/download/v${VERSION}/rk-core-${VERSION}-${TARGET}.tar.gz"

echo "Installing ReasonKit ${VERSION} for ${TARGET}..."

# Download and extract
curl -fsSL "$URL" | tar xz -C /tmp
sudo mv /tmp/rk-core /usr/local/bin/

echo "ReasonKit installed! Run 'rk-core --help' to get started."
```

---

## Release Checklist

### Pre-Release
- [ ] `cargo test` passes
- [ ] `cargo clippy -- -D warnings` clean
- [ ] `cargo fmt --check` passes
- [ ] CHANGELOG.md updated
- [ ] Version bumped

### Build
- [ ] Cross-compile all targets
- [ ] Generate shell completions
- [ ] Create SHA256 checksums
- [ ] Test binaries on each platform

### Distribution
- [ ] GitHub Release created
- [ ] `cargo publish`
- [ ] Homebrew formula updated
- [ ] Update install script

### Post-Release
- [ ] Announce release
- [ ] Monitor for issues
- [ ] Update documentation

---

## Essential Tools

```bash
# Core tools
cargo install cargo-release      # Version management
cargo install git-cliff          # Changelog generation
cargo install cargo-deny         # Security & licensing
cargo install cargo-audit        # Vulnerability scanning
cargo install cargo-binstall     # Fast binary installation
cargo install cross              # Cross-compilation

# Distribution
cargo install cargo-dist         # Release automation
cargo install cargo-deb          # Debian packages
cargo install cargo-bundle       # App bundling

# CI optimization
# Use Swatinem/rust-cache@v2 in GitHub Actions
```

---

## Quick Commands

```bash
# Release build (optimized)
cargo build --release

# Cross-compile for Linux (static)
cargo build --release --target x86_64-unknown-linux-musl

# Generate completions
cargo run --bin generate-completions

# Check security
cargo audit
cargo deny check

# Create changelog
git cliff -o CHANGELOG.md

# Release new version
cargo release patch --execute

# Publish to crates.io
cargo publish
```

---

## Package Manager Installation (User Perspective)

```bash
# Rust developers
cargo install reasonkit
cargo binstall reasonkit  # Faster, if binaries available

# macOS
brew install reasonkit

# Linux (Debian/Ubuntu)
wget https://github.com/user/reasonkit/releases/download/v0.1.0/reasonkit_0.1.0_amd64.deb
sudo dpkg -i reasonkit_0.1.0_amd64.deb

# Windows (Chocolatey)
choco install reasonkit

# Universal (install script)
curl -fsSL https://reasonkit.sh/install.sh | bash
```

---

## Key Insights

1. **Static linking (musl)** = Maximum Linux portability
2. **GitHub Actions + rust-cache** = Fast CI/CD
3. **cargo-dist** = Best release automation
4. **cargo-binstall** = Fast installation for users
5. **Shell completions** = Critical for UX
6. **Binary optimization** = Significantly impacts adoption
7. **Multi-channel distribution** = cargo install + Homebrew + GitHub Releases minimum

---

## Resources

- **Full Research:** [RUST_CLI_DEPLOYMENT_RESEARCH.md](./RUST_CLI_DEPLOYMENT_RESEARCH.md)
- **Rust CLI Book:** https://rust-cli.github.io/book/
- **Cargo Book:** https://doc.rust-lang.org/cargo/
- **cargo-dist:** https://opensource.axo.dev/cargo-dist/
- **git-cliff:** https://git-cliff.org/
- **cargo-deny:** https://embarkstudios.github.io/cargo-deny/

---

**Next Steps:** See Section 13 of full research for implementation roadmap.
