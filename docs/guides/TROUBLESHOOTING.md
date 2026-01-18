# Troubleshooting Guide

> **Quick fixes for the 10 most common issues**

---

## Top 10 Issues (Copy-Paste Fixes)

### 1. "command not found: rk"

**Fix:**

```bash
# Add cargo bin to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Make permanent (bash)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Make permanent (zsh)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

**Still not working?**

```bash
# Check if binary exists
ls -la ~/.cargo/bin/rk

# If not, reinstall
cargo install reasonkit-core --force
```

---

### 2. "API key not found"

**Fix:**

```bash
# Set API key (Anthropic)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $ANTHROPIC_API_KEY
```

**Make permanent:**

```bash
# Add to shell profile
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

---

### 3. "Rate limit exceeded (429)"

**Fix:**

```bash
# Use budget limit
rk think "query" --budget "$0.10"

# Use faster provider (Groq is cheap/fast)
export GROQ_API_KEY="gsk_..."
rk think "query" --provider groq

# Wait and retry
sleep 60 && rk think "query" --profile quick
```

---

### 4. "cargo: command not found" (Rust not installed)

**Fix:**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Activate
source "$HOME/.cargo/env"

# Verify
rustc --version
cargo --version

# Now install ReasonKit
cargo install reasonkit-core
```

---

### 5. "linker `cc` not found" (build error)

**Fix by platform:**

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt update && sudo apt install build-essential

# Fedora
sudo dnf groupinstall "Development Tools"

# Arch
sudo pacman -S base-devel
```

---

### 6. "OpenSSL not found" (build error)

**Fix by platform:**

```bash
# macOS
brew install openssl@3
export OPENSSL_DIR=$(brew --prefix openssl@3)

# Ubuntu/Debian
sudo apt install libssl-dev pkg-config

# Fedora
sudo dnf install openssl-devel

# Arch
sudo pacman -S openssl
```

---

### 7. "Timeout during analysis"

**Fix:**

```bash
# Use faster profile
rk think "query" --profile quick

# Increase timeout
rk think "query" --budget "5m"

# Use faster provider
rk think "query" --provider groq
```

---

### 8. "Output not JSON-parseable"

**Fix:**

```bash
# Ensure JSON format is specified
rk think "query" --format json

# Pipe to jq for validation
rk think "query" --format json | jq .

# If still failing, check for error messages in output
rk think "query" --format json 2>&1 | head -50
```

---

### 9. "Permission denied" during install

**Fix:**

```bash
# Don't use sudo with cargo
# This is correct:
cargo install reasonkit-core

# NOT this:
# sudo cargo install reasonkit-core  # WRONG

# If ~/.cargo/bin doesn't exist, create it
mkdir -p ~/.cargo/bin
```

---

### 10. "Low confidence scores" (<70%)

**This is expected behavior, not an error.**

**Improve confidence:**

```bash
# Use higher-rigor profile
rk think "query" --profile deep       # 85% target
rk think "query" --profile paranoid   # 95% target

# Use better model
rk think "query" --provider anthropic --model claude-opus-4

# Provide more context
rk think "Given context X, Y, Z, should we do A?" --profile balanced
```

---

## Diagnostic Commands

Run these to gather information for troubleshooting:

```bash
# Check installation
rk --version
which rk
echo $PATH | tr ':' '\n' | grep -E "(cargo|local)"

# Check Rust
rustc --version
cargo --version

# Check API keys
echo "ANTHROPIC: ${ANTHROPIC_API_KEY:0:10}..."
echo "OPENAI: ${OPENAI_API_KEY:0:10}..."

# Check system
uname -a
```

---

## Quick Reference: Common Fixes

| Problem                 | Quick Fix                               |
| ----------------------- | --------------------------------------- |
| `rk: command not found` | `export PATH="$HOME/.cargo/bin:$PATH"`  |
| `API key not found`     | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| `Rate limit exceeded`   | `rk think "..." --provider groq`        |
| `cargo not found`       | `curl -sSf https://sh.rustup.rs \| sh`  |
| `linker not found`      | `xcode-select --install` (macOS)        |
| `OpenSSL not found`     | `brew install openssl@3` (macOS)        |
| `Timeout`               | `--profile quick` or `--budget "5m"`    |
| `JSON parse error`      | `--format json` flag explicitly         |
| `Permission denied`     | Don't use `sudo` with cargo             |

---

## Still Stuck?

### 1. Check detailed installation docs

[INSTALLATION_TROUBLESHOOTING.md](INSTALLATION_TROUBLESHOOTING.md) - Platform-specific guides

### 2. Enable debug logging

```bash
RUST_LOG=debug rk think "query" --profile quick
```

### 3. Search existing issues

[GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues)

### 4. Ask the community

[GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions)

### 5. Open a new issue

Include:

- OS and version
- `rk --version` output
- Full error message
- Steps to reproduce

---

## Related Documentation

- [Installation Troubleshooting](INSTALLATION_TROUBLESHOOTING.md) - Deep platform-specific guides
- [Quickstart](QUICKSTART.md) - Getting started in 5 minutes
- [CLI Reference](CLI_REFERENCE.md) - Full command documentation

---

**Website:** [reasonkit.sh](https://reasonkit.sh) | **Docs:** [docs.rs/reasonkit-core](https://docs.rs/reasonkit-core)
