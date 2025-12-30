# API Versioning and Deprecation Strategy

> Comprehensive API lifecycle management for ReasonKit
> Enterprise-grade stability guarantees for CLI, Library, and REST APIs

**Version:** 1.0.0
**Last Updated:** 2025-12-28
**Status:** ACTIVE
**Applies To:** reasonkit-core (Rust CLI + Library), reasonkit (Python bindings), Future REST API

---

## Table of Contents

1. [Versioning Philosophy](#1-versioning-philosophy)
2. [CLI Versioning](#2-cli-versioning)
3. [Library API Versioning](#3-library-api-versioning)
4. [REST API Versioning (Future)](#4-rest-api-versioning-future)
5. [Deprecation Policy](#5-deprecation-policy)
6. [Breaking Change Process](#6-breaking-change-process)
7. [Compatibility Matrix](#7-compatibility-matrix)
8. [Release Process](#8-release-process)
9. [Enterprise Considerations](#9-enterprise-considerations)
10. [Migration Guide Template](#10-migration-guide-template)
11. [Version Lifecycle](#11-version-lifecycle)
12. [Communication Protocol](#12-communication-protocol)

---

## 1. Versioning Philosophy

### 1.1 Core Principles

```
"Stability is a feature, not an accident."

We prioritize:
1. Predictable evolution over rapid change
2. Clear communication over silent breakage
3. Smooth migration over forced upgrades
4. Enterprise reliability over bleeding-edge features
```

### 1.2 Semantic Versioning 2.0.0

ReasonKit strictly follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
  0.1.0           Development release (pre-stable)
  1.0.0           First stable release
  1.1.0           Backward-compatible features
  1.1.1           Backward-compatible bug fixes
  2.0.0           Breaking changes
  1.2.0-alpha.1   Pre-release (unstable)
  1.2.0-beta.2    Pre-release (feature complete)
  1.2.0-rc.1      Release candidate
```

### 1.3 Stability Tiers

| Version Range   | Stability         | API Guarantee  | Recommendation                   |
| --------------- | ----------------- | -------------- | -------------------------------- |
| **0.x.x**       | Development       | None           | Use with caution, expect changes |
| **1.0.0+**      | Stable            | Full SemVer    | Production use                   |
| **x.y.z-alpha** | Unstable          | None           | Testing only                     |
| **x.y.z-beta**  | Preview           | Feature-frozen | Early adopter testing            |
| **x.y.z-rc**    | Release Candidate | Near-final     | Final validation                 |

### 1.4 Stability Guarantees

#### Pre-1.0 (Current: v0.1.0)

```yaml
Pre-1.0 Stability Rules:
  - Minor versions (0.x.0) MAY include breaking changes
  - Patch versions (0.x.y) SHOULD be backward compatible
  - API stability is NOT guaranteed
  - Breaking changes documented in CHANGELOG.md
  - Migration guides provided for significant changes
```

#### Post-1.0 (Future)

```yaml
Post-1.0 Stability Rules:
  - SemVer strictly enforced
  - Public API frozen within major versions
  - Minimum 6-month deprecation notice
  - LTS versions with 2-year support
  - Security fixes backported to supported versions
```

### 1.5 What Constitutes Public API

```rust
// PUBLIC API (covered by stability guarantees)
pub struct Document { ... }           // Public structs
pub fn query(text: &str) -> Result    // Public functions
pub trait Storage { ... }             // Public traits
pub enum Error { ... }                // Public enums

// INTERNAL API (no stability guarantees)
pub(crate) fn internal_helper()       // Crate-internal
#[doc(hidden)] pub fn hidden()        // Hidden from docs
mod private { ... }                   // Private modules
```

**Rule:** If it's in `docs.rs` documentation, it's public API.

---

## 2. CLI Versioning

The command-line interface has distinct versioning considerations from the library API.

### 2.1 CLI Interface Classification

| Component              | Breaking              | Non-Breaking        | Notes                          |
| ---------------------- | --------------------- | ------------------- | ------------------------------ |
| **Commands**           | Remove, rename        | Add new commands    | Aliases preserve compatibility |
| **Subcommands**        | Remove, rename        | Add new subcommands | Aliases preserve compatibility |
| **Required Flags**     | Remove, rename        | Add optional flags  | New required flags = breaking  |
| **Optional Flags**     | Remove, rename        | Add, change default | Deprecation warnings first     |
| **Flag Values**        | Remove allowed values | Add new values      | Document all valid values      |
| **Output Format**      | Change structure      | Add fields          | JSON schema versioned          |
| **Exit Codes**         | Change meaning        | Add new codes       | Document all exit codes        |
| **Config File Format** | Schema changes        | Add optional fields | Version config format          |
| **Env Variables**      | Remove, rename        | Add new variables   | Document all variables         |
| **Default Behavior**   | Change defaults       | -                   | Major version only             |

### 2.2 CLI Version Display

```bash
# Version information
$ rk-core --version
rk-core 0.1.0

# Detailed version (for debugging)
$ rk-core --version --verbose
rk-core 0.1.0
  Commit: a1b2c3d
  Built: 2025-12-28
  Rust: 1.74.0
  Target: x86_64-unknown-linux-gnu
  Features: cli, local-embeddings
```

### 2.3 Breaking vs Non-Breaking Examples

#### Breaking Changes (Require MAJOR Bump)

```bash
# BREAKING: Renaming flag
# Before (0.x.x):
rk-core think --profile balanced "query"

# After (requires MAJOR):
rk-core think --mode balanced "query"  # --profile removed

# BREAKING: Changing required arguments
# Before:
rk-core query "search term"

# After (requires MAJOR):
rk-core query --text "search term"  # Positional now named

# BREAKING: Removing command
# Before:
rk-core export data.json

# After (requires MAJOR):
# Command removed without replacement

# BREAKING: Changing output format
# Before (JSON output):
{"results": [{"id": "123", "score": 0.95}]}

# After (requires MAJOR):
{"matches": [{"document_id": "123", "relevance": 0.95}]}

# BREAKING: Changing exit code meaning
# Before: exit 1 = general error
# After: exit 1 = file not found (requires MAJOR)
```

#### Non-Breaking Changes (MINOR Bump)

```bash
# NON-BREAKING: Adding new flag
rk-core query "search" --verbose    # New optional flag

# NON-BREAKING: Adding new command
rk-core analyze "document.pdf"      # New command

# NON-BREAKING: Adding alias
rk-core think --profile balanced    # Original
rk-core think -p balanced           # New alias

# NON-BREAKING: Adding new output field (JSON)
# Before:
{"results": [{"id": "123", "score": 0.95}]}

# After:
{"results": [{"id": "123", "score": 0.95, "latency_ms": 45}]}

# NON-BREAKING: Improving error messages
# Before: "Error: invalid input"
# After: "Error: invalid input - expected JSON, got XML"
```

### 2.4 Output Format Versioning

For machine-parseable output, version the output format:

```bash
# Request specific output version
rk-core query "term" --format json --format-version 1

# Output includes version
{
  "$schema": "https://reasonkit.sh/schemas/query-result/v1.json",
  "version": 1,
  "results": [...]
}
```

### 2.5 Configuration File Versioning

```toml
# ~/.reasonkit/config.toml
# Configuration format version (for migration)
config_version = 1

[storage]
backend = "qdrant"
url = "http://localhost:6333"

[embedding]
model = "text-embedding-3-small"
```

Migration between config versions:

```bash
# Auto-migrate config
rk-core config migrate --to 2

# Validate config format
rk-core config validate
```

---

## 3. Library API Versioning

### 3.1 Public API Definition

#### Rust Library (reasonkit-core)

```rust
// src/lib.rs - Public API surface

// PUBLIC: Exported in lib.rs, documented
pub mod thinktool;
pub mod retrieval;
pub mod embedding;
pub mod storage;

pub use types::{Document, Chunk, SearchResult};
pub use error::{Error, Result};

// INTERNAL: Not exported, no guarantees
mod internal;
pub(crate) mod utils;
```

#### Python Bindings (reasonkit)

```python
# PUBLIC: Listed in __all__, documented
from reasonkit import (
    Document,
    ThinkToolExecutor,
    query,
    embed,
)

# INTERNAL: Prefixed with underscore
from reasonkit._internal import _helper_function
```

### 3.2 API Stability Markers

```rust
/// Stable API - covered by SemVer guarantees
///
/// # Stability
/// This function is part of the stable API.
pub fn stable_function() -> Result<()> { ... }

/// Experimental API - may change without notice
///
/// # Stability
/// This is experimental and may change in minor versions.
/// Use feature flag: `experimental`
#[cfg(feature = "experimental")]
pub fn experimental_function() -> Result<()> { ... }

/// Deprecated API - will be removed
///
/// # Deprecated
/// Use `new_function` instead. Will be removed in 2.0.0.
#[deprecated(since = "1.1.0", note = "Use `new_function` instead")]
pub fn old_function() -> Result<()> { ... }
```

### 3.3 Breaking Changes Catalog

| Change Type                   | Breaking? | Mitigation                              |
| ----------------------------- | --------- | --------------------------------------- |
| **Remove public function**    | YES       | Deprecate first, provide alternative    |
| **Change function signature** | YES       | Create new function, deprecate old      |
| **Change return type**        | YES       | Version the return type or new function |
| **Add required parameter**    | YES       | Use builder pattern or options struct   |
| **Change error variants**     | YES       | Keep old variants, add new ones         |
| **Change struct fields**      | YES       | Use `#[non_exhaustive]` from start      |
| **Change trait definition**   | YES       | Create new trait version                |
| **Add optional parameter**    | NO        | Use `Option<T>` or default              |
| **Add new function**          | NO        | Just add it                             |
| **Add new error variant**     | Partial   | Use `#[non_exhaustive]`                 |
| **Add new struct field**      | Partial   | Use `#[non_exhaustive]`                 |
| **Improve performance**       | NO        | Should always be safe                   |
| **Fix bug**                   | Partial   | Document behavior change                |

### 3.4 Non-Exhaustive Types

For future-proof API design:

```rust
/// Document type - new variants may be added
#[non_exhaustive]
pub enum DocumentType {
    Paper,
    Documentation,
    Code,
    // Future: Transcript, Audio, Video
}

/// Search result - new fields may be added
#[non_exhaustive]
pub struct SearchResult {
    pub score: f32,
    pub document_id: Uuid,
    pub chunk: Chunk,
    // Future: match_positions, explanation
}
```

### 3.5 Builder Pattern for Extensibility

```rust
/// Extensible configuration without breaking changes
pub struct RetrievalConfig {
    top_k: usize,
    min_score: f32,
    alpha: f32,
    // Private fields can be added freely
}

impl RetrievalConfig {
    pub fn new() -> Self { ... }

    /// Add new options without breaking existing code
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// New option in v1.2 - doesn't break v1.1 code
    pub fn with_reranking(mut self, enabled: bool) -> Self {
        self.reranking = enabled;
        self
    }
}

// Usage remains backward compatible:
let config = RetrievalConfig::new()
    .with_top_k(10)
    .with_min_score(0.5);
```

### 3.6 Version Compatibility in Code

```rust
// Check version at runtime
const VERSION: &str = env!("CARGO_PKG_VERSION");

// Feature detection instead of version checking
#[cfg(feature = "local-embeddings")]
pub fn embed_locally(text: &str) -> Result<Vec<f32>> { ... }
```

---

## 4. REST API Versioning (Future)

When ReasonKit adds a REST API, these guidelines apply:

### 4.1 URL Path Versioning (Primary)

```
https://api.reasonkit.sh/v1/query
https://api.reasonkit.sh/v2/query

Benefits:
- Explicit version in every request
- Easy to route and proxy
- Clear in logs and documentation
- Works with any HTTP client
```

### 4.2 API Version Lifecycle

```
          ┌─────────────────────────────────────────────────────────────────────────┐
          │                          API VERSION LIFECYCLE                          │
          ├─────────────────────────────────────────────────────────────────────────┤
          │                                                                         │
          │  ACTIVE        DEPRECATED     SUNSET         RETIRED                    │
          │  ──────        ──────────     ──────         ───────                    │
          │                                                                         │
          │  ●────────────────●────────────────●────────────────●                   │
          │  │                │                │                │                   │
          │  │  Full support  │  Warnings      │  Read-only     │  Removed          │
          │  │  New features  │  No new feat   │  No writes     │  404 Not Found    │
          │  │  Bug fixes     │  Bug fixes     │  Bug fixes     │                   │
          │                                                                         │
          │  Timeline (from deprecation):                                           │
          │  ├─────────────────┼─────────────────┼─────────────────┤                │
          │  0                 6 months          12 months         (removal)         │
          │                                                                         │
          └─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Version Headers

```http
# Request specific version via header (secondary method)
GET /query HTTP/1.1
Host: api.reasonkit.sh
Accept: application/json
X-API-Version: 2024-12-01

# Response includes version info
HTTP/1.1 200 OK
X-API-Version: 2024-12-01
X-API-Deprecation: 2025-06-01
X-API-Sunset: 2025-12-01
Content-Type: application/json

{"results": [...]}
```

### 4.4 Deprecation Headers

```http
# Deprecation warning headers
Deprecation: @1735689600  # Unix timestamp of deprecation date
Sunset: Sat, 01 Jun 2025 00:00:00 GMT
Link: <https://docs.reasonkit.sh/api/migration/v1-to-v2>; rel="successor-version"
Warning: 299 - "API v1 is deprecated. Migrate to v2 by 2025-06-01"
```

### 4.5 Error Response for Retired Versions

```http
GET /v0/query HTTP/1.1
Host: api.reasonkit.sh

HTTP/1.1 410 Gone
Content-Type: application/json

{
  "error": "api_version_retired",
  "message": "API v0 was retired on 2024-06-01. Please upgrade to v2.",
  "documentation": "https://docs.reasonkit.sh/api/migration",
  "supported_versions": ["v1", "v2"],
  "latest_version": "v2"
}
```

### 4.6 Versioned Request/Response Schemas

```json
// GET /v1/query
// Response Schema v1
{
  "$schema": "https://reasonkit.sh/schemas/api/v1/query-response.json",
  "results": [
    {
      "id": "doc_123",
      "score": 0.95,
      "text": "..."
    }
  ]
}

// GET /v2/query
// Response Schema v2 (new structure)
{
  "$schema": "https://reasonkit.sh/schemas/api/v2/query-response.json",
  "data": {
    "matches": [
      {
        "document_id": "doc_123",
        "relevance_score": 0.95,
        "content": "...",
        "metadata": { ... }
      }
    ]
  },
  "meta": {
    "request_id": "req_abc",
    "latency_ms": 45
  }
}
```

---

## 5. Deprecation Policy

### 5.1 Deprecation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DEPRECATION TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Phase 1: ANNOUNCEMENT (Day 0)                                                  │
│  ───────────────────────────────                                                │
│  - Feature marked as deprecated in code                                         │
│  - CHANGELOG entry under "Deprecated"                                           │
│  - Migration guide published                                                    │
│  - Blog post / announcement (if significant)                                    │
│                                                                                 │
│  Phase 2: WARNING PERIOD (6 months)                                             │
│  ──────────────────────────────────                                             │
│  - Runtime deprecation warnings emitted                                         │
│  - Documentation shows deprecated badge                                         │
│  - Alternative clearly documented                                               │
│  - Enterprise customers notified directly                                       │
│                                                                                 │
│  Phase 3: FINAL NOTICE (3 months before removal)                                │
│  ───────────────────────────────────────────────                                │
│  - Loud warnings in logs/output                                                 │
│  - Email to known enterprise users                                              │
│  - GitHub issue for tracking                                                    │
│                                                                                 │
│  Phase 4: REMOVAL (Next major version)                                          │
│  ──────────────────────────────────────                                         │
│  - Feature removed from codebase                                                │
│  - Clean error if deprecated feature is used                                    │
│  - Migration guide still available                                              │
│                                                                                 │
│  Total Minimum Deprecation Period: 6 months (stable), 2 minor versions (pre-1.0)│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Deprecation in Code

#### Rust

````rust
/// Query the knowledge base.
///
/// # Deprecated
/// This function is deprecated since v1.2.0.
/// Use [`search`] instead for improved performance.
///
/// # Migration
/// ```rust
/// // Before:
/// let results = query("text", 10)?;
///
/// // After:
/// let results = search(SearchQuery::new("text").with_limit(10))?;
/// ```
#[deprecated(
    since = "1.2.0",
    note = "Use `search` instead. Will be removed in 2.0.0. See migration guide: https://docs.reasonkit.sh/migration/query-to-search"
)]
pub fn query(text: &str, limit: usize) -> Result<Vec<SearchResult>> {
    // Log deprecation warning at runtime
    tracing::warn!(
        "Deprecated function `query` called. Migrate to `search` before v2.0.0. \
         Migration guide: https://docs.reasonkit.sh/migration/query-to-search"
    );

    // Delegate to new implementation
    search(SearchQuery::new(text).with_limit(limit))
}
````

#### Python

```python
import warnings
from functools import wraps

def deprecated(version: str, replacement: str, removal: str):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated since {version}. "
                f"Use {replacement} instead. "
                f"Will be removed in {removal}. "
                f"See: https://docs.reasonkit.sh/migration/{func.__name__}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated(version="1.2.0", replacement="search()", removal="2.0.0")
def query(text: str, limit: int = 10):
    """[DEPRECATED] Query the knowledge base."""
    return search(SearchQuery(text).with_limit(limit))
```

#### CLI

```rust
// Deprecated CLI flag handling
if args.deprecated_flag.is_some() {
    eprintln!(
        "Warning: --deprecated-flag is deprecated since v1.2.0.\n\
         Use --new-flag instead.\n\
         Will be removed in v2.0.0.\n\
         Migration: https://docs.reasonkit.sh/cli/migration/deprecated-flag"
    );
}
```

### 5.3 Deprecation Documentation

Every deprecated feature must have:

````markdown
## Deprecated: `query()` Function

**Deprecated in:** v1.2.0
**Removal planned:** v2.0.0 (estimated Q3 2025)
**Replacement:** `search()` function

### Why This Change?

The `query()` function had limited flexibility. The new `search()` function
provides:

- Builder pattern for configuration
- Better performance (2x faster)
- More result metadata

### Migration Steps

1. Replace `query(text, limit)` with `search(SearchQuery::new(text).with_limit(limit))`
2. Update error handling for new error types
3. Test with new response structure

### Before/After Examples

```rust
// BEFORE (v1.1 and earlier)
let results = query("machine learning", 10)?;
for result in results {
    println!("{}: {}", result.id, result.score);
}

// AFTER (v1.2+)
let results = search(
    SearchQuery::new("machine learning")
        .with_limit(10)
        .with_min_score(0.5)
)?;
for result in results {
    println!("{}: {} ({}ms)", result.id, result.score, result.latency_ms);
}
```
````

````

### 5.4 Deprecation Tracking

```yaml
# .github/DEPRECATION_REGISTRY.yaml
deprecated_features:
  - name: "query function"
    deprecated_in: "1.2.0"
    removal_target: "2.0.0"
    replacement: "search function"
    migration_guide: "docs/migration/query-to-search.md"
    tracking_issue: "#123"
    enterprise_notified: true

  - name: "--profile CLI flag"
    deprecated_in: "1.3.0"
    removal_target: "2.0.0"
    replacement: "--reasoning-mode"
    migration_guide: "docs/cli/migration/profile-flag.md"
    tracking_issue: "#456"
    enterprise_notified: false
````

---

## 6. Breaking Change Process

### 6.1 RFC Process for Breaking Changes

Major breaking changes require an RFC (Request for Comments):

```markdown
# RFC: Restructure Search API

## Summary

Proposal to replace `query()` with a more flexible `search()` API.

## Motivation

- Current API lacks flexibility for advanced queries
- Performance bottleneck in result formatting
- Customer feedback requesting more metadata

## Detailed Design

[Technical specification...]

## Drawbacks

- Migration effort for existing users
- Potential for bugs during transition

## Alternatives Considered

1. Extend existing `query()` - rejected due to signature complexity
2. New `query_v2()` - rejected, confusing naming

## Migration Path

[Step-by-step migration guide...]

## Timeline

- RFC Open: 2025-01-15
- RFC Close: 2025-02-15 (30 days)
- Implementation: v1.3.0
- Deprecation: v1.3.0
- Removal: v2.0.0
```

### 6.2 Breaking Change Checklist

Before introducing a breaking change:

```markdown
## Breaking Change Checklist

### Documentation

- [ ] RFC document created and approved
- [ ] Migration guide written
- [ ] API documentation updated
- [ ] CHANGELOG entry drafted

### Code

- [ ] New API implemented and tested
- [ ] Old API wrapped with deprecation warning
- [ ] Deprecation period (6 months) scheduled
- [ ] Feature flag for early testing (optional)

### Communication

- [ ] Changelog entry under "Breaking Changes"
- [ ] Migration guide published
- [ ] Blog post drafted (for major changes)
- [ ] Enterprise customers notified directly
- [ ] GitHub issue created for tracking

### Testing

- [ ] Migration tested on sample projects
- [ ] Performance comparison documented
- [ ] Integration tests updated

### Compatibility

- [ ] Backward compatibility shim available (if possible)
- [ ] Version compatibility matrix updated
- [ ] Installation instructions updated

### Timeline

- [ ] Deprecation announced: \_\_\_\_
- [ ] Warning period begins: \_\_\_\_
- [ ] Final notice: \_\_\_\_
- [ ] Removal in version: \_\_\_\_
```

### 6.3 Breaking Change Categories

| Category                       | Severity | Deprecation Period | Enterprise Notice |
| ------------------------------ | -------- | ------------------ | ----------------- |
| **Core API** (Document, Query) | High     | 12 months          | Required          |
| **CLI Commands**               | High     | 6 months           | Required          |
| **Configuration Format**       | Medium   | 6 months           | Required          |
| **Output Format**              | Medium   | 6 months           | Email + docs      |
| **Error Types**                | Low      | 3 months           | Docs only         |
| **Internal APIs**              | None     | N/A                | None              |

---

## 7. Compatibility Matrix

### 7.1 Version Compatibility

| reasonkit-core | Rust MSRV | Python Bindings | Protocol Version | Status  |
| -------------- | --------- | --------------- | ---------------- | ------- |
| 0.1.x          | 1.74+     | 3.9+            | 1.0              | Current |
| 0.2.x          | 1.75+     | 3.9+            | 1.0              | Planned |
| 1.0.x          | TBD       | 3.10+           | 1.0              | Future  |
| 1.1.x          | TBD       | 3.10+           | 1.1              | Future  |

**MSRV Policy:** Minimum Supported Rust Version is bumped only in MINOR versions, never in PATCH.

### 7.2 Python Version Support

| Python Version | Status    | Support Until        |
| -------------- | --------- | -------------------- |
| 3.9            | Supported | 2025-10 (Python EOL) |
| 3.10           | Supported | 2026-10              |
| 3.11           | Supported | 2027-10              |
| 3.12           | Supported | 2028-10              |
| 3.13           | Supported | 2029-10              |

**Policy:** Support Python versions for their upstream support lifetime.

### 7.3 Operating System Support

| OS      | Versions       | Architecture    | Tier   |
| ------- | -------------- | --------------- | ------ |
| Linux   | glibc 2.17+    | x86_64, aarch64 | Tier 1 |
| Linux   | musl 1.1.24+   | x86_64, aarch64 | Tier 1 |
| macOS   | 12+ (Monterey) | x86_64, aarch64 | Tier 1 |
| Windows | 10, 11         | x86_64          | Tier 1 |
| FreeBSD | 13+            | x86_64          | Tier 2 |

**Tier 1:** Official binaries, tested in CI
**Tier 2:** Should work, limited testing

### 7.4 Protocol Version Compatibility

| Protocol Version | reasonkit-core | Features                   |
| ---------------- | -------------- | -------------------------- |
| 1.0              | 0.1.0+         | Base ThinkTools, RAG       |
| 1.1              | 1.0.0+         | Extended profiles, budgets |
| 2.0              | 2.0.0+         | New step types, streaming  |

**Protocol files (YAML)** have their own version:

```yaml
# protocol-v1.0
version: "1.0"
name: gigathink
steps:
  - ...
```

### 7.5 Dependency Compatibility

```toml
# Key dependencies and their version ranges
[dependencies]
tokio = "1.20+"       # Async runtime
serde = "1.0"         # Serialization (very stable)
qdrant-client = "1.5+" # Vector database
clap = "4.0+"         # CLI parsing

# Dependencies with breaking change potential
pyo3 = "0.22"         # Python bindings - tightly versioned
```

---

## 8. Release Process

### 8.1 Release Cadence

| Release Type      | Frequency      | Content                    | Notice    |
| ----------------- | -------------- | -------------------------- | --------- |
| **Patch** (x.y.Z) | As needed      | Bug fixes only             | Immediate |
| **Minor** (x.Y.0) | Monthly        | New features, deprecations | 1 week    |
| **Major** (X.0.0) | Annually (max) | Breaking changes           | 3 months  |

### 8.2 Release Checklist

```markdown
## Release v1.2.0 Checklist

### Pre-Release

- [ ] All quality gates passing (CONS-009)
  - [ ] `cargo build --release`
  - [ ] `cargo clippy -- -D warnings`
  - [ ] `cargo fmt --check`
  - [ ] `cargo test --all-features`
  - [ ] `cargo audit`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Migration guide ready (if breaking changes)
- [ ] Documentation updated

### Release

- [ ] Create git tag: `git tag -a v1.2.0 -m "Release v1.2.0"`
- [ ] Push tag: `git push origin v1.2.0`
- [ ] Verify CI/CD pipeline success
- [ ] GitHub Release created with notes

### Post-Release

- [ ] Verify crates.io publication
- [ ] Verify PyPI publication
- [ ] Verify npm publication
- [ ] Test installation from all channels
- [ ] Update website version badge
- [ ] Send enterprise notifications (if applicable)
- [ ] Post release announcement
```

### 8.3 Hotfix Process

For critical bug fixes in stable versions:

```bash
# Create hotfix branch from tag
git checkout v1.1.0
git checkout -b hotfix/1.1.1

# Apply fix
# ... fix the bug ...

# Bump patch version
# Cargo.toml: version = "1.1.1"

# Commit and tag
git commit -am "fix: critical security issue in auth module"
git tag -a v1.1.1 -m "Hotfix v1.1.1 - Security fix"
git push origin hotfix/1.1.1 --tags

# Also apply fix to main
git checkout main
git cherry-pick <commit-hash>
```

---

## 9. Enterprise Considerations

### 9.1 Long-Term Support (LTS) Versions

Post-1.0, designated LTS versions receive extended support:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LTS VERSION SUPPORT                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Version    │ Release    │ Active Support │ Security Only │ End of Life         │
│  ───────────┼────────────┼────────────────┼───────────────┼───────────────────  │
│  1.0.x LTS  │ 2025-06    │ 2026-06        │ 2027-06       │ 2027-06             │
│  2.0.x      │ 2026-06    │ 2027-06        │ 2027-12       │ 2027-12             │
│  2.4.x LTS  │ 2027-01    │ 2028-01        │ 2029-01       │ 2029-01             │
│                                                                                 │
│  LTS Designation: Every 4th minor version (1.0, 1.4, 2.0, 2.4, ...)            │
│  Active Support: 12 months (features + bugs + security)                         │
│  Security Only: 12 months (critical security fixes only)                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Backport Policy

| Fix Type          | Backport to LTS | Backport to Previous |
| ----------------- | --------------- | -------------------- |
| Security Critical | Yes             | Yes (30 days)        |
| Security High     | Yes             | Best effort          |
| Bug Critical      | Yes             | No                   |
| Bug Major         | Best effort     | No                   |
| Bug Minor         | No              | No                   |

### 9.3 Extended Deprecation for Enterprise

Enterprise customers with support contracts may request:

- **Extended Deprecation Period:** Up to 18 months (vs 6 months standard)
- **Dedicated Migration Support:** Direct access to engineering team
- **Custom Builds:** Version with specific deprecated features retained
- **Private Security Notices:** 48-hour advance notice of security advisories

### 9.4 Enterprise Notification Process

```yaml
Enterprise Breaking Change Notification:
  Day -90: Email to enterprise contacts
          Subject: "ReasonKit: Breaking change planned for v2.0.0"
          Content: Summary, migration guide link, support options

  Day -60: Follow-up email with migration resources
          Subject: "ReasonKit: Migration resources for v2.0.0"

  Day -30: Final reminder
          Subject: "ReasonKit: v2.0.0 releases in 30 days"

  Day 0:   Release announcement
          Subject: "ReasonKit v2.0.0 Released"

  Day +14: Check-in email
          Subject: "ReasonKit: How is your v2.0.0 migration going?"
```

### 9.5 Custom Support Agreements

Enterprise customers may negotiate:

- **Private Builds:** Specific versions with custom patches
- **Accelerated Fixes:** Priority bug fix development
- **Consultation Hours:** Direct access to architects for migration planning
- **Testing Environments:** Early access to pre-release versions
- **Rollback Assistance:** Help reverting to previous versions if needed

---

## 10. Migration Guide Template

Use this template for all breaking changes:

````markdown
# Migration Guide: [Feature Name] from v[OLD] to v[NEW]

## Overview

This guide helps you migrate from [old feature/API] to [new feature/API].

**Affected versions:** v[OLD].x
**Target version:** v[NEW].0+
**Estimated effort:** [1 hour / 2-4 hours / 1 day / 1 week]
**Difficulty:** [Low / Medium / High]

## What Changed?

### Before (v[OLD])

```rust
// Old API usage
```
````

### After (v[NEW])

```rust
// New API usage
```

### Why This Change?

[Explain the motivation for the change]

## Step-by-Step Migration

### Step 1: Update Dependencies

```toml
# Cargo.toml
reasonkit-core = "[NEW]"
```

### Step 2: Find Affected Code

Search for usages of [old feature]:

```bash
# Find all usages
grep -r "old_function" src/
```

### Step 3: Replace Usages

[Specific replacement instructions]

### Step 4: Handle Edge Cases

[Document any edge cases]

### Step 5: Test Changes

```bash
cargo test
```

## Common Issues

### Issue 1: [Description]

**Solution:** [How to fix]

### Issue 2: [Description]

**Solution:** [How to fix]

## Compatibility Shim (Optional)

If you need more time to migrate, use this compatibility shim:

```rust
// Temporary compatibility - remove before v[NEXT]
mod compat {
    pub fn old_function(...) {
        new_function(...)
    }
}
```

## Need Help?

- Documentation: [link]
- GitHub Issues: [link]
- Discord: [link]
- Enterprise Support: support@reasonkit.sh

```

---

## 11. Version Lifecycle

### 11.1 Version States

```

┌─────────────────────────────────────────────────────────────────────────────────┐
│ VERSION LIFECYCLE │
├─────────────────────────────────────────────────────────────────────────────────┤
│ │
│ DEVELOPMENT │
│ │ │
│ ▼ │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ ALPHA │────▶│ BETA │────▶│ RC │────▶│ STABLE │ │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
│ - Unstable - Features - Bug fixes - Full support │
│ - Testing - Stabilizing - Final test - SemVer rules │
│ │
│ │ │
│ ▼ │
│ ┌───────────┐ │
│ │DEPRECATED │ │
│ └───────────┘ │
│ - Warning │
│ - Security only │
│ │ │
│ ▼ │
│ ┌───────────┐ │
│ │ EOL │ │
│ └───────────┘ │
│ - No support │
│ - No updates │
│ │
└─────────────────────────────────────────────────────────────────────────────────┘

````

### 11.2 Support Level by State

| State | Bug Fixes | Security Fixes | Features | Documentation |
|-------|-----------|----------------|----------|---------------|
| Alpha | No | No | Active | Draft |
| Beta | Critical only | Yes | Frozen | Draft |
| RC | Critical only | Yes | Frozen | Final |
| Stable | Yes | Yes | Next minor | Maintained |
| Deprecated | Critical only | Yes | No | Archived |
| EOL | No | No | No | Archived |

---

## 12. Communication Protocol

### 12.1 Communication Channels

| Change Type | CHANGELOG | Blog | Email | GitHub | Twitter |
|-------------|-----------|------|-------|--------|---------|
| Patch release | Yes | No | No | Release | No |
| Minor release | Yes | Optional | Enterprise | Release | Optional |
| Major release | Yes | Yes | All | Release | Yes |
| Security fix | Yes | Yes | All | Advisory | Yes |
| Deprecation | Yes | Optional | Enterprise | Issue | Optional |

### 12.2 Changelog Format

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

## [Unreleased]

### Added
- New `search()` function with builder pattern (#123)

### Changed
- Improved error messages in CLI output

### Deprecated
- `query()` function - use `search()` instead (#124)

### Removed
- (Only in major versions)

### Fixed
- Memory leak in embedding cache (#125)

### Security
- Updated dependency to fix CVE-2025-XXXX

## [1.2.0] - 2025-01-15

### Added
...
````

### 12.3 Release Notes Format

````markdown
# ReasonKit Core v1.2.0 Release Notes

## Highlights

- **New Search API:** More flexible and 2x faster
- **Budget Management:** Control costs with time/token/cost budgets
- **Improved CLI:** Better error messages and new `--verbose` flag

## Breaking Changes

None in this release.

## Deprecations

- `query()` function is deprecated. Use `search()` instead.
  Migration guide: https://docs.reasonkit.sh/migration/query-to-search
  Removal planned: v2.0.0

## New Features

### New Search API

The new `search()` function provides...

## Bug Fixes

- Fixed memory leak in embedding cache (#125)
- Corrected CLI output formatting (#126)

## Upgrade Instructions

```bash
cargo install reasonkit-core --version 1.2.0
# or
pip install --upgrade reasonkit==1.2.0
```
````

## Full Changelog

https://github.com/reasonkit/reasonkit-core/compare/v1.1.0...v1.2.0

```

---

## Summary

This API Versioning and Deprecation Strategy ensures:

1. **Predictable Evolution:** Semantic versioning strictly followed post-1.0
2. **Clear Communication:** Deprecation warnings, migration guides, timely notifications
3. **Enterprise Reliability:** LTS versions, extended support, custom agreements
4. **Smooth Migration:** Minimum 6-month deprecation periods with comprehensive guides
5. **Multiple API Types:** Separate strategies for CLI, Library, and REST APIs
6. **Comprehensive Documentation:** Every change documented with migration paths

---

## Related Documents

- [VERSIONING.md](VERSIONING.md) - Basic version policy
- [RELEASE_PROCESS.md](RELEASE_PROCESS.md) - Release workflow
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) - Pre-release verification
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [API_REFERENCE.md](API_REFERENCE.md) - Public API documentation

---

*ReasonKit Core - API Versioning Strategy v1.0.0*
*https://reasonkit.sh*
*Last Updated: 2025-12-28*
```
