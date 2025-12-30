# Verification Module - Protocol Delta

**Status:** Production Ready
**Version:** 1.0.0
**Tests:** 21/21 passing

---

## Overview

The verification module implements **Protocol Delta's ProofLedger** - an immutable citation ledger that uses cryptographic binding to prevent research drift and ensure claims remain verifiable over time.

## Philosophy

> "We do not quote the wind. We quote the stone."

Traditional citations use URLs as weak pointers that can break or change. Protocol Delta uses content-addressed storage with SHA-256 hashing to create immutable anchors.

---

## Quick Start

```rust
use reasonkit_core::verification::ProofLedger;

// Create ledger
let ledger = ProofLedger::new("./research.db")?;

// Anchor a claim
let content = "The global AI market was valued at $196.63B in 2023.";
let hash = ledger.anchor(content, "https://example.com", None)?;

// Later: verify it hasn't changed
let result = ledger.verify(&hash, content)?;
assert!(result.verified);
```

---

## Features

- **SHA-256 Content Hashing** - Cryptographically secure content binding
- **SQLite Ledger** - Persistent, ACID-compliant storage
- **Drift Detection** - Automatically detect when cited content changes
- **Metadata Support** - Attach JSON metadata to each anchor
- **Query by URL** - Find all citations from a given source
- **Timestamp Tracking** - Know exactly when content was anchored
- **Unicode Support** - Full international character support

---

## Use Cases

### 1. Academic Research

```rust
// Anchor findings from papers
let hash = ledger.anchor(
    "RAPTOR improves QuALITY by 20%",
    "https://arxiv.org/abs/2401.18059",
    Some(r#"{"type": "benchmark", "confidence": 0.95}"#.to_string())
)?;
```

### 2. Fact Checking

```rust
// Verify a claim hasn't drifted
let original = ledger.get_anchor(&hash)?;
let current = fetch_current_content(&original.url)?;
let result = ledger.verify(&hash, &current)?;

if !result.verified {
    eprintln!("WARNING: Content has changed since citation!");
}
```

### 3. Audit Trail

```rust
// List all citations from a source
let anchors = ledger.list_by_url("https://arxiv.org/abs/2401.18059")?;

for anchor in anchors {
    println!("Anchored: {}", anchor.timestamp);
    println!("Content: {}", anchor.content_snippet);
}
```

---

## API Reference

### `ProofLedger`

#### Constructors

- `new(path)` - Create persistent ledger at path
- `in_memory()` - Create temporary in-memory ledger

#### Methods

- `anchor(content, url, metadata)` - Create immutable anchor, returns hash
- `get_anchor(hash)` - Retrieve anchor by hash
- `verify(hash, content)` - Check if content matches anchor
- `check_drift(hash, refetched)` - Verify refetched content
- `list_by_url(url)` - Get all anchors from a URL
- `count()` - Total anchors in ledger

### `Anchor`

Fields:

- `hash: String` - SHA-256 hash (64 hex chars)
- `url: String` - Source URL
- `timestamp: DateTime<Utc>` - When anchored
- `content_snippet: String` - First 200 chars
- `metadata: Option<String>` - JSON metadata

### `VerificationResult`

Fields:

- `verified: bool` - Whether verification passed
- `original_hash: String` - Hash from anchor
- `current_hash: String` - Hash of current content
- `message: String` - Human-readable result
- `anchor: Anchor` - Original anchor data

---

## Citation Format

**Traditional:**

```text
The market grew by 5% [1].
[1] https://finance.yahoo.com/article/123
```

**Protocol Delta:**

```text
The market grew by 5% [1].
[1] sha256:8f4a1c2b... (Verified 2025-12-23)
    → https://finance.yahoo.com/article/123
```

---

## Database Schema

```sql
CREATE TABLE anchors (
    hash TEXT PRIMARY KEY,          -- SHA-256 of content
    url TEXT NOT NULL,              -- Source URL
    timestamp TEXT NOT NULL,        -- ISO 8601
    content_snippet TEXT NOT NULL,  -- First 200 chars
    metadata TEXT                   -- JSON metadata
);

CREATE INDEX idx_anchors_url ON anchors(url);
CREATE INDEX idx_anchors_timestamp ON anchors(timestamp);
```

---

## Performance

- **Hash computation:** O(n) where n = content length
- **Anchor creation:** O(1) indexed insert
- **Verification:** O(n) + O(1) lookup
- **URL query:** O(log n) via B-tree index

Typical operations:

- Anchor 1KB content: ~0.1ms
- Verify content: ~0.1ms
- Query by URL: ~0.01ms

---

## Examples

Run the demo:

```bash
cargo run --example protocol_delta_demo
```

See full examples in:

- `examples/protocol_delta_demo.rs` - Complete workflow
- `tests/verification_integration.rs` - Test scenarios

---

## Testing

```bash
# Unit tests (10)
cargo test --lib verification

# Integration tests (11)
cargo test --test verification_integration

# All tests
cargo test verification
```

---

## Future Work

### Visual Fetch Integration

When the Python `visual_fetch.py` component is integrated, it will:

1. Render pages with browser automation
2. Use Vision Language Models to extract content
3. Pass extracted content to ProofLedger for anchoring
4. Enable citations from dynamic/visual content

---

## Dependencies

- `sha2` - SHA-256 hashing
- `rusqlite` - SQLite database
- `chrono` - Timestamps
- `serde`, `serde_json` - Serialization
- `thiserror` - Error handling

---

## License

Apache 2.0 (same as reasonkit-core)

---

## References

- **Design Spec:** `/research/protocol_delta/PROTOCOL_DELTA.md`
- **Integration Doc:** `/reasonkit-core/PROTOCOL_DELTA_INTEGRATION.md`
- **Module Docs:** Run `cargo doc --open` and navigate to `reasonkit_core::verification`

---

_Part of Protocol Delta - The Anchor_
_ReasonKit Core v1.0.0_
