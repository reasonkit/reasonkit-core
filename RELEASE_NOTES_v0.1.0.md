# ReasonKit Core v0.1.0 Release Notes

**Release Date:** December 2025
**License:** Apache 2.0
**Minimum Rust Version:** 1.74

---

## Overview

ReasonKit Core v0.1.0 is the initial public release of the AI Thinking Enhancement System. This release establishes the foundation for structured AI reasoning through the ThinkTools protocol architecture.

**Tagline:** Turn Prompts into Protocols

---

## Core Features

### ThinkTools (5 Reasoning Modules)

| Module            | Shortcut | Purpose                                            |
| ----------------- | -------- | -------------------------------------------------- |
| **GigaThink**     | `gt`     | Multi-perspective expansion (10+ viewpoints)       |
| **LaserLogic**    | `ll`     | Precision deductive reasoning, fallacy detection   |
| **BedRock**       | `br`     | First principles decomposition, axiom rebuilding   |
| **ProofGuard**    | `pg`     | Multi-source verification, contradiction detection |
| **BrutalHonesty** | `bh`     | Adversarial self-critique, find flaws first        |

### Reasoning Profiles

```bash
rk think --profile quick "Fast 3-step analysis"
rk think --profile balanced "Standard 5-module chain"
rk think --profile deep "Thorough analysis with all modules"
rk think --profile paranoid "Maximum verification mode"
rk think --profile scientific "Research and experiments"
```

### CLI Commands

```bash
# Core reasoning
rk think --profile <profile> "<query>"

# Deep research
rk web --depth <level> "<query>"

# RAG operations
rk query "<query>"
rk ingest <file>

# Verification
rk anchor --url <url> "<content>"
rk verify --hash <hash> "<content>"

# Trace management
rk trace list
rk trace export <id>
```

---

## Infrastructure

### Retrieval System

- **Hybrid Search:** Vector + BM25 combination
- **RRF Fusion:** Reciprocal Rank Fusion for result combining
- **Query Expansion:** Automatic query variant generation
- **Cross-Encoder Reranking:** Precision improvement

### Storage

- **Qdrant Integration:** Vector database support
- **Tantivy BM25:** Full-text search indexing
- **In-Memory Mode:** Development and testing

### Embedding Support

- OpenAI embeddings (ada-002, text-embedding-3-small/large)
- Local embeddings via ONNX (optional feature)

### MCP Server

- Protocol Delta tools for verification
- Anchor/verify/lookup operations

---

## Quality Gates Passed

| Gate   | Command                       | Status              |
| ------ | ----------------------------- | ------------------- |
| Build  | `cargo build --release`       | ✅ PASS             |
| Lint   | `cargo clippy -- -D warnings` | ✅ PASS             |
| Format | `cargo fmt --check`           | ✅ PASS             |
| Tests  | `cargo test --lib`            | ✅ 187 tests passed |

---

## Technical Details

### Dependencies (Key)

- `qdrant-client` 1.10 - Vector database
- `tantivy` 0.22 - Full-text search
- `tokio` 1.x - Async runtime
- `reqwest` 0.12 - HTTP client
- `serde` 1.0 - Serialization

### Optional Features

- `local-embeddings` - BGE-M3 ONNX inference
- `arf` - Autonomous Reasoning Framework
- `embedded-qdrant` - Embedded mode

### Performance Targets

- Core operations: < 5ms latency
- Reranking: < 200ms for 20 candidates
- Memory efficient for large documents

---

## Breaking Changes

N/A - Initial release

---

## Known Issues

1. Integration tests need API sync with library changes
2. `--all-features` requires Rust-bert dependency fixes
3. Workspace profile configuration warning (non-blocking)

---

## Upgrade Path

N/A - Initial release

---

## Contributors

- ReasonKit Team

---

## Documentation

- [Quick Reference](docs/thinktools/THINKTOOLS_QUICK_REFERENCE.md)
- [CLI Workflow Examples](docs/guides/CLI_WORKFLOW_EXAMPLES.md)
- [API Reference](docs/reference/API_REFERENCE.md)

---

## Installation

```bash
# From source
cargo install --path .

# Or via release binary
curl -fsSL https://get.reasonkit.sh | bash
```

---

_"Designed, Not Dreamed. Turn Prompts into Protocols."_
_<https://reasonkit.sh>_
