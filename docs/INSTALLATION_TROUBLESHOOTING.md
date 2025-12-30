# Installation Troubleshooting Guide

> Comprehensive troubleshooting for ReasonKit Core installation issues
> Covers all platforms, installation methods, and common failure modes

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Platform-Specific Issues](#platform-specific-issues)
   - [macOS](#macos-intel--apple-silicon)
   - [Linux](#linux-ubuntu-debian-fedora-arch)
   - [Windows](#windows-native--wsl)
3. [Common Installation Errors](#common-installation-errors)
4. [Dependency Issues](#dependency-issues)
5. [Network Issues](#network-issues)
6. [Post-Installation Issues](#post-installation-issues)
7. [Verification Steps](#verification-steps)
8. [Getting Help](#getting-help)

---

## Quick Diagnostics

Run this diagnostic script to identify common issues:

```bash
#!/bin/bash
echo "=== ReasonKit Installation Diagnostics ==="

echo -e "\n[1] Checking Rust installation..."
if command -v rustc &>/dev/null; then
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
else
    echo "  ERROR: Rust not found. Install from https://rustup.rs"
fi

echo -e "\n[2] Checking PATH..."
echo "  PATH contains ~/.cargo/bin: $(echo $PATH | grep -q '.cargo/bin' && echo 'YES' || echo 'NO')"
echo "  PATH contains ~/.local/bin: $(echo $PATH | grep -q '.local/bin' && echo 'YES' || echo 'NO')"

echo -e "\n[3] Checking rk-core binary..."
if command -v rk-core &>/dev/null; then
    echo "  Location: $(which rk-core)"
    echo "  Version: $(rk-core --version 2>/dev/null || echo 'Unable to get version')"
else
    echo "  ERROR: rk-core not found in PATH"
fi

echo -e "\n[4] Checking dependencies..."
pkg-config --version &>/dev/null && echo "  pkg-config: installed" || echo "  pkg-config: NOT FOUND"
openssl version &>/dev/null && echo "  OpenSSL: $(openssl version)" || echo "  OpenSSL: NOT FOUND"

echo -e "\n[5] Checking system..."
echo "  OS: $(uname -s)"
echo "  Arch: $(uname -m)"
echo "  Shell: $SHELL"
```

---

## Platform-Specific Issues

### macOS (Intel + Apple Silicon)

#### Issue: Xcode Command Line Tools Not Installed

**Error Message:**

```
error: linker `cc` not found
```

or

```
xcrun: error: invalid active developer path
```

**Cause:** Rust requires C compiler and linker from Xcode Command Line Tools.

**Solution:**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# If already installed but not working, reset
sudo xcode-select --reset

# Verify installation
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

**Prevention:** Always run `xcode-select --install` before installing Rust on a fresh macOS.

---

#### Issue: OpenSSL Not Found on macOS

**Error Message:**

```
error: failed to run custom build command for `openssl-sys`
Could not find directory of OpenSSL installation
```

**Cause:** macOS no longer ships with OpenSSL headers. Homebrew OpenSSL is not automatically linked.

**Solution:**

```bash
# Install OpenSSL via Homebrew
brew install openssl@3

# Set environment variables for build (add to ~/.zshrc or ~/.bashrc)
export OPENSSL_DIR=$(brew --prefix openssl@3)
export PKG_CONFIG_PATH="$OPENSSL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"

# Retry installation
cargo install reasonkit-core
```

**For Apple Silicon (M1/M2/M3):**

```bash
# Homebrew on Apple Silicon uses /opt/homebrew
export OPENSSL_DIR="/opt/homebrew/opt/openssl@3"
export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@3/lib/pkgconfig:$PKG_CONFIG_PATH"
```

**Prevention:** Add the export statements to your shell profile permanently.

---

#### Issue: Apple Silicon Architecture Mismatch

**Error Message:**

```
error[E0463]: can't find crate for `std`
note: the `aarch64-apple-darwin` target may not be installed
```

or

```
ld: building for macOS-arm64 but attempting to link with file built for macOS-x86_64
```

**Cause:** Rust toolchain installed for wrong architecture, or mixing x86_64 and arm64 binaries.

**Solution:**

```bash
# Check current architecture
uname -m
# Should output: arm64 (for Apple Silicon) or x86_64 (for Intel)

# Reinstall Rust for correct architecture
rustup self uninstall  # Remove existing installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify correct target
rustup show
# Should list "aarch64-apple-darwin" for Apple Silicon

# If running in Rosetta (x86_64 emulation), exit and use native terminal
arch  # Shows current execution architecture
```

**Prevention:** Ensure Terminal is running natively, not under Rosetta.

---

### Linux (Ubuntu, Debian, Fedora, Arch)

#### Issue: Build Essentials Missing (Ubuntu/Debian)

**Error Message:**

```
error: linker `cc` not found
```

or

```
error: could not compile `ring` due to previous error
```

**Cause:** C compiler (gcc/clang) and essential build tools not installed.

**Solution (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install build-essential pkg-config libssl-dev
```

**Solution (Fedora):**

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install pkg-config openssl-devel
```

**Solution (Arch):**

```bash
sudo pacman -S base-devel pkg-config openssl
```

**Prevention:** Install build essentials before any Rust development.

---

#### Issue: libssl-dev / openssl-devel Missing

**Error Message:**

```
error: failed to run custom build command for `openssl-sys`
Could not find openssl via pkg-config
```

or

```
Header openssl/ssl.h not found
```

**Cause:** OpenSSL development headers not installed.

**Solution (Ubuntu/Debian):**

```bash
sudo apt install libssl-dev
```

**Solution (Fedora/RHEL/CentOS):**

```bash
sudo dnf install openssl-devel
# or for older systems
sudo yum install openssl-devel
```

**Solution (Arch):**

```bash
sudo pacman -S openssl
```

**Solution (Alpine):**

```bash
apk add openssl-dev musl-dev
```

**Prevention:** Include libssl-dev in your initial system setup.

---

#### Issue: pkg-config Not Found

**Error Message:**

```
error: could not find system library 'openssl' required by the 'openssl-sys' crate
```

or

```
pkg-config not found
```

**Cause:** pkg-config utility is missing.

**Solution (Ubuntu/Debian):**

```bash
sudo apt install pkg-config
```

**Solution (Fedora):**

```bash
sudo dnf install pkgconf-pkg-config
```

**Solution (Arch):**

```bash
sudo pacman -S pkgconf
```

**Prevention:** Include pkg-config in build essentials installation.

---

#### Issue: GLIBC Version Too Old

**Error Message:**

```
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```

**Cause:** Pre-built binary requires newer glibc than your system provides.

**Solution:**

```bash
# Check your glibc version
ldd --version

# Option 1: Build from source instead of using pre-built binary
cargo install reasonkit-core --locked

# Option 2: Update your distribution
sudo apt update && sudo apt upgrade

# Option 3: Use container with newer base
docker run -it ubuntu:22.04
```

**Prevention:** Use `cargo install` to build from source on older systems.

---

### Windows (Native + WSL)

#### Issue: MSVC Build Tools Missing

**Error Message:**

```
error: linker `link.exe` not found
```

or

```
error: could not find `link.exe`
```

**Cause:** Visual Studio Build Tools not installed.

**Solution:**

```powershell
# Option 1: Install Visual Studio Build Tools (recommended)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++" workload

# Option 2: Install via winget
winget install Microsoft.VisualStudio.2022.BuildTools

# After installation, restart terminal and retry
cargo install reasonkit-core
```

**Prevention:** Install Build Tools before Rust on Windows.

---

#### Issue: Rust Not Using Correct Linker on Windows

**Error Message:**

```
error: linking with `link.exe` failed
```

with various sub-errors about missing libraries.

**Cause:** Rust cannot find MSVC linker or wrong toolchain selected.

**Solution:**

```powershell
# Check current default toolchain
rustup show

# Ensure MSVC toolchain is default
rustup default stable-msvc

# If using x64 system, ensure x64 target
rustup target add x86_64-pc-windows-msvc
```

**Prevention:** Use `rustup default stable-msvc` immediately after Rust installation on Windows.

---

#### Issue: Long Path Names on Windows

**Error Message:**

```
error: failed to open file: The system cannot find the path specified
```

or compilation fails with path-related errors.

**Cause:** Windows MAX_PATH (260 character) limitation.

**Solution:**

```powershell
# Enable long paths (requires admin PowerShell)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use shorter installation path
# Clone to C:\rk instead of deep nested directories
cd C:\
git clone https://github.com/reasonkit/reasonkit-core rk
cd rk && cargo build --release
```

**Prevention:** Enable long paths or use short directory names.

---

#### Issue: WSL2 Performance / File System

**Error Message:**
Compilation is extremely slow, or:

```
error: could not write to file
```

**Cause:** Compiling on Windows filesystem from WSL is slow. Or WSL disk is full.

**Solution:**

```bash
# Work within Linux filesystem (not /mnt/c/)
cd ~
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core
cargo build --release

# Check disk space
df -h /

# If WSL disk is full, clean up
cargo clean
docker system prune -a  # if using Docker in WSL
```

**Prevention:** Always work in `/home/` directory in WSL, not `/mnt/c/`.

---

## Common Installation Errors

### Error: Rust/Cargo Not Found

**Error Message:**

```bash
cargo: command not found
```

or

```bash
rustc: command not found
```

**Cause:** Rust is not installed, or PATH is not configured.

**Solution:**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the environment (or restart terminal)
source "$HOME/.cargo/env"

# Verify installation
rustc --version
cargo --version
```

**Prevention:** Always run `source "$HOME/.cargo/env"` after installing Rust, or restart your terminal.

---

### Error: Permission Denied

**Error Message:**

```
error: could not write to /usr/local/bin/rk-core
Permission denied (os error 13)
```

**Cause:** Attempting to install to system directory without sudo, or incorrect permissions.

**Solution:**

```bash
# Use user-local installation (recommended)
cargo install reasonkit-core
# Installs to ~/.cargo/bin/

# Ensure ~/.cargo/bin is in PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Alternative: Use custom install directory
CARGO_INSTALL_ROOT="$HOME/.local" cargo install reasonkit-core
# Installs to ~/.local/bin/
```

**Prevention:** Never use `sudo cargo install`. Use user-local paths.

---

### Error: PATH Not Set

**Error Message:**

```bash
rk-core: command not found
```

(after successful installation)

**Cause:** Installation directory not in PATH.

**Solution:**

```bash
# For cargo install (default location: ~/.cargo/bin)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# For curl installer (default location: ~/.local/bin)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# For Zsh users
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# For Fish users
fish_add_path ~/.cargo/bin
```

**Prevention:** Add PATH export to shell profile during initial setup.

---

### Error: Version Conflicts

**Error Message:**

```
error: failed to select a version for the requirement `tokio = "^1"`
```

or

```
error[E0308]: mismatched types (version conflict between dependencies)
```

**Cause:** Conflicting dependency versions, often from outdated Cargo.lock.

**Solution:**

```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean

# Update dependencies
cargo update

# Retry installation
cargo install reasonkit-core --force

# If building from source
git pull
cargo update
cargo build --release
```

**Prevention:** Keep Rust toolchain updated with `rustup update` regularly.

---

### Error: Cargo Install Timeout

**Error Message:**

```
error: failed to download `reasonkit-core`
Timeout while downloading
```

**Cause:** Slow network, or crates.io temporarily unavailable.

**Solution:**

```bash
# Set longer timeout
CARGO_HTTP_TIMEOUT=120 cargo install reasonkit-core

# Use alternative registry mirror (if available)
# Add to ~/.cargo/config.toml:
# [source.crates-io]
# replace-with = "ustc"
# [source.ustc]
# registry = "https://mirrors.ustc.edu.cn/crates.io-index/"

# Alternative: clone and build locally
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core
cargo build --release
cp target/release/rk-core ~/.local/bin/
```

**Prevention:** Use local build for unreliable networks.

---

### Error: Out of Memory During Compilation

**Error Message:**

```
error: could not compile `reasonkit-core`
Killed
```

(compilation terminates abruptly)

**Cause:** Insufficient RAM for compilation (especially with LTO enabled).

**Solution:**

```bash
# Reduce parallelism
CARGO_BUILD_JOBS=1 cargo install reasonkit-core

# Or limit codegen units in Cargo.toml (if building from source)
# [profile.release]
# codegen-units = 4  # instead of 1

# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Prevention:** Ensure at least 4GB RAM + 4GB swap for compilation with LTO.

---

## Dependency Issues

### libssl-dev / OpenSSL Development Headers

See [Platform-Specific Issues](#issue-libssl-dev--openssl-devel-missing) above.

---

### pkg-config Issues

See [Platform-Specific Issues](#issue-pkg-config-not-found) above.

---

### C Compiler Missing

**Error Message:**

```
error: linker `cc` not found
```

**Cause:** No C compiler installed.

**Solution by Platform:**

| Platform      | Command                                     |
| ------------- | ------------------------------------------- |
| Ubuntu/Debian | `sudo apt install build-essential`          |
| Fedora        | `sudo dnf groupinstall "Development Tools"` |
| Arch          | `sudo pacman -S base-devel`                 |
| macOS         | `xcode-select --install`                    |
| Windows       | Install Visual Studio Build Tools           |

---

### CMake Not Found (for certain optional features)

**Error Message:**

```
error: failed to run custom build command for `ring`
CMake not found
```

**Cause:** Some dependencies require CMake for building.

**Solution:**

```bash
# Ubuntu/Debian
sudo apt install cmake

# Fedora
sudo dnf install cmake

# macOS
brew install cmake

# Windows
winget install Kitware.CMake
```

---

### Perl Not Found (for OpenSSL build)

**Error Message:**

```
error: failed to run custom build command for `openssl-sys`
Can't locate... (Perl error)
```

**Cause:** Perl is required to build OpenSSL from source.

**Solution:**

```bash
# Ubuntu/Debian
sudo apt install perl

# Fedora
sudo dnf install perl

# macOS (usually pre-installed)
brew install perl

# Windows
# Install Strawberry Perl: https://strawberryperl.com/
```

---

## Network Issues

### Corporate Proxy Configuration

**Error Message:**

```
error: failed to download from `https://crates.io/`
```

or

```
error: unable to get local issuer certificate
```

**Cause:** Corporate proxy blocking or intercepting HTTPS traffic.

**Solution:**

```bash
# Set proxy environment variables
export http_proxy="http://proxy.company.com:8080"
export https_proxy="http://proxy.company.com:8080"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# For cargo specifically
# Add to ~/.cargo/config.toml:
[http]
proxy = "http://proxy.company.com:8080"

# If proxy uses SSL interception (corporate MITM)
# Add to ~/.cargo/config.toml:
[http]
check-revoke = false
cainfo = "/path/to/corporate-ca-bundle.crt"

# Alternative: Build from source downloaded manually
# Download tarball from GitHub releases and build offline
```

**Prevention:** Configure proxy in shell profile and cargo config.

---

### Firewall Blocking crates.io

**Error Message:**

```
error: failed to download `reasonkit-core`
Could not resolve host: crates.io
```

**Cause:** Firewall blocking access to crates.io or GitHub.

**Solution:**

```bash
# Test connectivity
curl -I https://crates.io
curl -I https://github.com

# If blocked, use alternative download:
# 1. Download source on machine with internet access
git clone https://github.com/reasonkit/reasonkit-core
tar -czvf reasonkit-core.tar.gz reasonkit-core

# 2. Transfer to restricted machine
scp reasonkit-core.tar.gz user@restricted:/tmp/

# 3. Build on restricted machine
tar -xzvf reasonkit-core.tar.gz
cd reasonkit-core
cargo build --release --offline  # Uses vendored dependencies if available
```

---

### DNS Resolution Failures

**Error Message:**

```
error: failed to resolve: `crates.io`
```

**Cause:** DNS server not responding or misconfigured.

**Solution:**

```bash
# Test DNS
nslookup crates.io
dig crates.io

# Try alternative DNS
# Add to /etc/resolv.conf:
nameserver 8.8.8.8
nameserver 1.1.1.1

# Or use systemd-resolved
sudo systemctl restart systemd-resolved
```

---

### SSL/TLS Certificate Errors

**Error Message:**

```
error: [60] SSL certificate problem: unable to get local issuer certificate
```

**Cause:** CA certificates outdated or missing.

**Solution:**

```bash
# Ubuntu/Debian
sudo apt install ca-certificates
sudo update-ca-certificates

# Fedora
sudo dnf install ca-certificates
sudo update-ca-trust

# macOS
# Usually handled by system, but if needed:
brew install ca-certificates

# Set certificate bundle path
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

---

## Post-Installation Issues

### Binary Runs But Crashes Immediately

**Error Message:**

```
Segmentation fault (core dumped)
```

or

```
Illegal instruction
```

**Cause:** Binary compiled for different CPU architecture or features.

**Solution:**

```bash
# Rebuild from source with compatible settings
RUSTFLAGS="-C target-cpu=native" cargo install reasonkit-core --force

# Or use generic build
RUSTFLAGS="-C target-cpu=generic" cargo install reasonkit-core --force
```

---

### Configuration File Not Found

**Error Message:**

```
Error: Config file not found at ~/.config/reasonkit/config.toml
```

**Cause:** First run without configuration file.

**Solution:**

```bash
# Create default configuration
mkdir -p ~/.config/reasonkit

cat > ~/.config/reasonkit/config.toml << 'EOF'
[general]
log_level = "info"

[thinktool]
default_provider = "anthropic"
default_profile = "balanced"
EOF

# Verify
rk-core doctor check
```

---

### API Key Environment Variable Not Set

**Error Message:**

```
Error: API key not found. Set ANTHROPIC_API_KEY environment variable.
```

**Cause:** LLM provider API key not configured.

**Solution:**

```bash
# Set temporarily
export ANTHROPIC_API_KEY="sk-ant-..."

# Set permanently (bash)
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc

# Set permanently (zsh)
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
source ~/.zshrc

# Or use .env file in project directory
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

---

## Verification Steps

### How to Verify Successful Installation

```bash
# 1. Check binary exists and runs
rk-core --version
# Expected: reasonkit-core 0.1.0

# 2. Check help displays correctly
rk-core --help
# Should display command options

# 3. Check configuration is valid
rk-core doctor check
# Should show health status

# 4. Test basic functionality (requires API key)
export ANTHROPIC_API_KEY="sk-ant-..."
rk-core think "What is 2+2?" --profile quick
# Should return structured response
```

### Quick Functionality Test

```bash
# Test without API key (mock mode)
rk-core think "Test query" --mock --profile quick
# Should return simulated response

# Test JSON output
rk-core think "Test" --mock --format json | jq .
# Should output valid JSON
```

### Version Information

```bash
# Full version details
rk-core --version

# Check Rust version used for build
rk-core doctor check --verbose
```

---

## Getting Help

### Before Opening an Issue

1. **Check this guide** - Most issues are covered above
2. **Run diagnostics** - Use the script at the top of this document
3. **Check existing issues** - Search [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues)
4. **Update and retry** - `rustup update && cargo install reasonkit-core --force`

### GitHub Issue Template

When opening an issue, include:

```markdown
## Environment

- **OS:** [e.g., Ubuntu 22.04, macOS 14.0, Windows 11]
- **Architecture:** [e.g., x86_64, aarch64/Apple Silicon]
- **Rust version:** `rustc --version`
- **Cargo version:** `cargo --version`
- **Installation method:** [cargo install / curl script / source build]

## Error
```

[Paste exact error message here]

```

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [See error]

## Attempted Solutions
- [ ] Ran `rustup update`
- [ ] Installed system dependencies (list which)
- [ ] Tried building from source
- [ ] Checked PATH configuration

## Additional Context
[Any other relevant information]
```

### Community Resources

| Resource           | URL                                                                                                        | Purpose                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------ |
| GitHub Discussions | [github.com/reasonkit/reasonkit-core/discussions](https://github.com/reasonkit/reasonkit-core/discussions) | Q&A, feature requests    |
| GitHub Issues      | [github.com/reasonkit/reasonkit-core/issues](https://github.com/reasonkit/reasonkit-core/issues)           | Bug reports              |
| Discord            | [discord.gg/reasonkit](https://discord.gg/reasonkit)                                                       | Real-time community help |
| Documentation      | [reasonkit.sh/docs](https://reasonkit.sh/docs)                                                             | Official docs            |

### Support Channels

1. **Community Support** (free)
   - GitHub Discussions for Q&A
   - Discord for real-time help

2. **Enterprise Support** (reasonkit-pro license)
   - Priority email support
   - Dedicated Slack channel
   - Video call troubleshooting

---

## Quick Reference: Common Fixes

| Problem                      | Quick Fix                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `cargo: command not found`   | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh && source ~/.cargo/env` |
| `rk-core: command not found` | `export PATH="$HOME/.cargo/bin:$PATH"`                                                   |
| OpenSSL errors (macOS)       | `brew install openssl@3 && export OPENSSL_DIR=$(brew --prefix openssl@3)`                |
| OpenSSL errors (Linux)       | `sudo apt install libssl-dev` or `sudo dnf install openssl-devel`                        |
| linker not found             | `sudo apt install build-essential` or `xcode-select --install`                           |
| pkg-config not found         | `sudo apt install pkg-config`                                                            |
| Permission denied            | Use `cargo install` (installs to ~/.cargo/bin, no sudo)                                  |
| Compilation killed (OOM)     | `CARGO_BUILD_JOBS=1 cargo install reasonkit-core`                                        |
| Proxy issues                 | `export https_proxy="http://proxy:port"`                                                 |

---

## Installation Methods Summary

| Method               | Command                                           | Best For                       |
| -------------------- | ------------------------------------------------- | ------------------------------ |
| **Cargo Install**    | `cargo install reasonkit-core`                    | Rust developers, most reliable |
| **Universal Script** | `curl -fsSL https://reasonkit.sh/install \| bash` | Quick setup, auto-detection    |
| **From Source**      | `git clone ... && cargo build --release`          | Development, customization     |
| **npm**              | `npm install -g @reasonkit/cli`                   | Node.js ecosystem (wrapper)    |
| **pip**              | `pip install reasonkit`                           | Python ecosystem (bindings)    |

---

_Document Version: 1.0.0_
_Last Updated: 2025-12-28_
_Maintainer: ReasonKit Team_
