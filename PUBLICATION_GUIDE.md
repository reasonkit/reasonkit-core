# Crates.io Publication Guide

## Overview

This guide covers the publication workflow for `reasonkit-mem` and `reasonkit-core` to crates.io.

## Current Status

### ✅ reasonkit-mem v0.1.0

- **Status:** Ready to publish
- **Dry-run:** ✅ Successful
- **Blocker:** Requires crates.io credentials

### ⚠️ reasonkit-core v1.0.1

- **Status:** Blocked on reasonkit-mem publication
- **Blocker:** Path dependency must be converted to version dependency
- **Dry-run:** ❌ Fails (dependency issue)

---

## Publication Workflow

### Step 1: Publish reasonkit-mem

```bash
cd reasonkit-mem

# Verify readiness
cargo publish --dry-run

# Authenticate with crates.io (if not already done)
cargo login <your-crates-io-token>

# Publish
cargo publish
```

**After successful publication:**

- Wait 5-10 minutes for crates.io index to update
- Verify: `cargo search reasonkit-mem` should show v0.1.0

---

### Step 2: Update reasonkit-core Cargo.toml

**File:** `reasonkit-core/Cargo.toml`

**Change:**

```toml
# BEFORE (path dependency - blocks publication)
reasonkit-mem = { path = "../reasonkit-mem", optional = true }

# AFTER (version dependency - ready for publication)
reasonkit-mem = { version = "0.1.0", optional = true }
```

**Quick fix command:**

```bash
cd reasonkit-core
sed -i 's|reasonkit-mem = { path = "../reasonkit-mem", optional = true }|reasonkit-mem = { version = "0.1.0", optional = true }|' Cargo.toml
```

---

### Step 3: Verify reasonkit-core

```bash
cd reasonkit-core

# Update dependencies
cargo update

# Verify build
cargo build --release

# Verify tests
cargo test --all-features

# Dry-run publication
cargo publish --dry-run
```

---

### Step 4: Publish reasonkit-core

```bash
cd reasonkit-core

# Authenticate (if not already done)
cargo login <your-crates-io-token>

# Publish
cargo publish
```

---

## Prerequisites

### 1. Crates.io Account

- Create account at https://crates.io
- Verify email address

### 2. API Token

- Go to https://crates.io/me
- Generate new token
- Run: `cargo login <token>`

### 3. Git Clean State

```bash
# Ensure all changes are committed
git status

# If uncommitted changes exist:
git add .
git commit -m "Prepare for crates.io publication"
```

---

## Verification Checklist

### Pre-Publication (reasonkit-mem)

- [ ] `cargo publish --dry-run` succeeds
- [ ] All tests pass: `cargo test --all-features`
- [ ] Linting passes: `cargo clippy -- -D warnings`
- [ ] Formatting passes: `cargo fmt --check`
- [ ] Git is clean: `git status`
- [ ] Version number is correct: `0.1.0`
- [ ] README.md is present and complete
- [ ] LICENSE file is present (Apache-2.0)

### Pre-Publication (reasonkit-core)

- [ ] `reasonkit-mem` is published and available
- [ ] `Cargo.toml` uses version dependency (not path)
- [ ] `cargo update` succeeds
- [ ] `cargo build --release` succeeds
- [ ] `cargo test --all-features` passes
- [ ] `cargo publish --dry-run` succeeds
- [ ] Git is clean: `git status`
- [ ] Version number is correct: `1.0.1`

---

## Post-Publication

### Verify Publication

```bash
# Check reasonkit-mem
cargo search reasonkit-mem

# Check reasonkit-core
cargo search reasonkit-core

# Verify installation
cargo install reasonkit-core
```

### Update Documentation

- [ ] Update README.md with crates.io badges
- [ ] Update installation instructions
- [ ] Update website/docs with publication status

---

## Troubleshooting

### Error: "all dependencies must have a version requirement"

**Solution:** Convert path dependencies to version dependencies before publishing.

### Error: "failed to verify manifest"

**Solution:** Check Cargo.toml syntax, ensure all required fields are present.

### Error: "crate already exists"

**Solution:** Version already published. Increment version in Cargo.toml.

### Error: "uncommitted changes"

**Solution:** Commit all changes before publishing.

---

## Notes

- **Version Numbers:** Follow semantic versioning (MAJOR.MINOR.PATCH)
- **Index Update:** crates.io index updates every 5-10 minutes
- **Yanking:** If needed, yank with: `cargo yank --vers <version> <crate-name>`
- **Dependencies:** All dependencies must be published or available on crates.io

---

**Last Updated:** 2025-12-30
**Status:** Ready for publication (requires crates.io credentials)
