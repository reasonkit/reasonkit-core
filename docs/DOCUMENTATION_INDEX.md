# ReasonKit Core Documentation Index

> Central hub for all ReasonKit Core documentation

**Version:** 0.1.0
**License:** Apache 2.0
**Last Updated:** 2025-12-23

---

## Quick Links

| Document | Purpose | Audience |
|----------|---------|----------|
| [API Reference](API_REFERENCE.md) | Complete API documentation | Developers |
| [CLI Reference](CLI_REFERENCE.md) | Command-line interface guide | Users & DevOps |
| [CLI Workflow Examples](CLI_WORKFLOW_EXAMPLES.md) | Practical workflow scripts | Users |
| [ThinkTools Quick Reference](THINKTOOLS_QUICK_REFERENCE.md) | Protocol execution guide | AI Engineers |
| [ThinkTools Architecture](THINKTOOLS_ARCHITECTURE.md) | System design & internals | Architects |
| [Architecture](../ARCHITECTURE.md) | 5-layer RAG design | Engineers |
| [Project Context](../CLAUDE.md) | Project-specific details | Contributors |

---

## Documentation Categories

### For Users

**Getting Started**
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](CLI_REFERENCE.md) - Complete CLI guide

**Using ReasonKit**
- [Query Knowledge Base](#querying)
- [Ingest Documents](#ingestion)
- [Execute ThinkTools](#thinktools)
- [CLI Workflow Examples](CLI_WORKFLOW_EXAMPLES.md) - Research, decisions, code review
- [Example Scripts](../scripts/examples/) - Practical bash scripts
- [Configuration](#configuration)

### For Developers

**API Documentation**
- [API Reference](API_REFERENCE.md) - Complete API docs
- [Rustdoc](../target/doc/reasonkit_core/index.html) - Generated API docs
- [Core Types](#core-types)
- [Modules](#modules)

**Architecture**
- [System Architecture](../ARCHITECTURE.md) - 5-layer design
- [ThinkTools Architecture](THINKTOOLS_ARCHITECTURE.md) - Protocol engine
- [Storage Backends](#storage)
- [Retrieval System](#retrieval)

**Contributing**
- [Development Setup](#development)
- [Testing](#testing)
- [Quality Gates](#quality-gates)

### For Researchers

**Academic References**
- [Paper Index](content/papers-index.md)
- [Source Overview](SOURCE_OVERVIEW.md)
- [Research Documentation](research/)

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core
cargo install --path .
```

### From Cargo (when published)

```bash
cargo install reasonkit-core
```

### Verify Installation

```bash
rk-core --version
# reasonkit-core 0.1.0
```

---

## Quick Start

### 1. Setup

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Start Qdrant (optional - can use in-memory)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Ingest Documents

```bash
# Ingest PDF papers
rk-core ingest ./papers --doc-type paper --recursive

# Verify
rk-core stats
```

### 3. Query

```bash
# Semantic search
rk-core query "What is chain-of-thought reasoning?"

# Hybrid search (semantic + keyword)
rk-core query "transformer architecture" --hybrid --top-k 10
```

### 4. Execute ThinkTools

```bash
# Quick analysis
rk-core think "Should we adopt microservices?" --profile quick

# Deep analysis with verification
rk-core think "Evaluate this architecture" --profile paranoid
```

---

## Core Types

### Document

Represents a document in the knowledge base.

```rust
pub struct Document {
    pub id: Uuid,
    pub doc_type: DocumentType,
    pub source: Source,
    pub content: DocumentContent,
    pub metadata: Metadata,
    pub processing: ProcessingStatus,
    pub chunks: Vec<Chunk>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}
```

**See:** [API Reference - Document](API_REFERENCE.md#document)

### Chunk

A segmented piece of text optimized for retrieval.

```rust
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub index: usize,
    pub section: Option<String>,
    pub page: Option<usize>,
    pub embedding_ids: EmbeddingIds,
}
```

**See:** [API Reference - Chunk](API_REFERENCE.md#chunk)

### SearchResult

Search result with relevance scoring.

```rust
pub struct SearchResult {
    pub score: f32,
    pub document_id: Uuid,
    pub chunk: Chunk,
    pub match_source: MatchSource,
}
```

**See:** [API Reference - SearchResult](API_REFERENCE.md#searchresult)

---

## Modules

### Core Modules

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `lib.rs` | Core types & exports | [API Ref](API_REFERENCE.md#core-types) |
| `error.rs` | Error handling | [API Ref](API_REFERENCE.md#error-handling) |
| `ingestion/` | Document parsing | [API Ref](API_REFERENCE.md#ingestion) |
| `processing/` | Chunking & cleaning | [API Ref](API_REFERENCE.md#processing) |
| `embedding/` | Text embeddings | [API Ref](API_REFERENCE.md#embedding-pipeline) |
| `indexing/` | BM25 & HNSW indexes | [API Ref](API_REFERENCE.md#indexing) |
| `retrieval/` | Hybrid search | [API Ref](API_REFERENCE.md#retrieval-system) |
| `storage/` | Qdrant & storage | [API Ref](API_REFERENCE.md#storage-backends) |
| `raptor.rs` | RAPTOR trees | [API Ref](API_REFERENCE.md#raptor-hierarchical-retrieval) |

### ThinkTool Modules

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `thinktool/mod.rs` | Protocol engine | [ThinkTools Arch](THINKTOOLS_ARCHITECTURE.md) |
| `thinktool/llm.rs` | LLM integrations (18+ providers) | [API Ref](API_REFERENCE.md#unifiedllmclient) |
| `thinktool/executor.rs` | Protocol execution | [ThinkTools Quick](THINKTOOLS_QUICK_REFERENCE.md) |
| `thinktool/registry.rs` | Protocol registry | [ThinkTools Arch](THINKTOOLS_ARCHITECTURE.md) |
| `thinktool/profiles.rs` | Reasoning profiles | [API Ref](API_REFERENCE.md#reasoning-profiles) |
| `thinktool/budget.rs` | Budget management | [API Ref](API_REFERENCE.md#budget-management) |
| `thinktool/trace.rs` | Execution tracing | [API Ref](API_REFERENCE.md#execution-trace) |

---

## Querying

### Basic Query

```bash
rk-core query "your question here"
```

### Hybrid Search (Recommended)

Combines semantic (vector) and keyword (BM25) search:

```bash
rk-core query "transformer architecture" --hybrid --top-k 10
```

### RAPTOR Tree Retrieval

For long documents with hierarchical structure:

```bash
rk-core query "explain the full methodology" --raptor
```

### Output Formats

```bash
# Human-readable text (default)
rk-core query "query" --format text

# JSON for scripting
rk-core query "query" --format json > results.json

# Markdown
rk-core query "query" --format markdown
```

**See:** [CLI Reference - Query](CLI_REFERENCE.md#2-query---search-knowledge-base)

---

## Ingestion

### Ingest Single File

```bash
rk-core ingest paper.pdf --doc-type paper
```

### Ingest Directory

```bash
rk-core ingest ./papers --recursive --doc-type paper
```

### Supported Formats

- **PDF** - Academic papers, reports
- **Markdown** - Documentation, notes
- **HTML** - Web pages
- **JSON/JSONL** - Structured data

### Document Types

- `paper` - Academic papers
- `documentation` - Technical docs
- `code` - Source code
- `note` - Personal notes
- `transcript` - Meeting/interview transcripts
- `benchmark` - Benchmark data

**See:** [CLI Reference - Ingest](CLI_REFERENCE.md#1-ingest---ingest-documents)

---

## ThinkTools

### What are ThinkTools?

Structured reasoning protocols that transform ad-hoc LLM prompting into auditable, reproducible reasoning chains.

### Available Protocols (Open Source)

| Tool | Code | Purpose |
|------|------|---------|
| **GigaThink** | `gt` | Expansive creative thinking (10+ perspectives) |
| **LaserLogic** | `ll` | Precision deductive reasoning, fallacy detection |
| **BedRock** | `br` | First principles decomposition |
| **ProofGuard** | `pg` | Multi-source verification |
| **BrutalHonesty** | `bh` | Adversarial self-critique |

### Reasoning Profiles

| Profile | Modules | Confidence | Use Case |
|---------|---------|------------|----------|
| `--quick` | gt, ll | 70% | Fast 3-step analysis |
| `--balanced` | gt, ll, br, pg | 80% | Standard 5-module chain |
| `--deep` | gt, ll, br, pg, hr | 85% | Thorough analysis |
| `--paranoid` | gt, ll, br, pg, bh | 95% | Maximum verification |

### Execute Protocol

```bash
# Specific protocol
rk-core think "Should we adopt microservices?" --protocol gigathink

# Profile (multiple protocols)
rk-core think "Evaluate this decision" --profile balanced
```

### Supported LLM Providers (18+)

- **Anthropic** (Claude Opus/Sonnet/Haiku)
- **OpenAI** (GPT-4, GPT-3.5)
- **Google Gemini** (Gemini 2.0 Pro/Flash)
- **xAI** (Grok-2)
- **Groq** (Ultra-fast inference)
- **OpenRouter** (300+ models)
- **DeepSeek** (DeepSeek-V3, DeepSeek-R1)
- **Mistral AI** (Mistral Large, Codestral)
- And 10+ more...

**See:**
- [ThinkTools Quick Reference](THINKTOOLS_QUICK_REFERENCE.md)
- [ThinkTools Architecture](THINKTOOLS_ARCHITECTURE.md)
- [CLI Reference - Think](CLI_REFERENCE.md#3-think---execute-thinktool-protocols)

---

## Configuration

### Configuration File

Create `~/.reasonkit/config.toml`:

```toml
[storage]
backend = "qdrant"
qdrant_url = "http://localhost:6333"

[embedding]
model = "text-embedding-3-small"
dimension = 1536
batch_size = 100

[retrieval]
top_k = 10
alpha = 0.7
fusion_strategy = "rrf"

[thinktool]
default_provider = "anthropic"
default_model = "claude-sonnet-4"
save_traces = true
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export QDRANT_API_KEY="..."

# Paths
export REASONKIT_DATA_DIR="./data"
export REASONKIT_CONFIG="~/.reasonkit/config.toml"

# LLM Providers
export OPENROUTER_API_KEY="sk-or-..."
export GROQ_API_KEY="gsk_..."
export XAI_API_KEY="xai-..."
```

**See:** [CLI Reference - Environment Variables](CLI_REFERENCE.md#environment-variables)

---

## Storage

### Backends

| Backend | Use Case | Setup |
|---------|----------|-------|
| **Qdrant** | Production | `docker run -p 6333:6333 qdrant/qdrant` |
| **In-Memory** | Testing | No setup required |
| **JSONL** | Lightweight | File-based persistence |

### Qdrant Setup

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Configure
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-api-key"
```

### In-Memory (No Setup)

```bash
# Automatically used if Qdrant not available
rk-core ingest ./papers --recursive
```

**See:** [API Reference - Storage Backends](API_REFERENCE.md#storage-backends)

---

## Retrieval

### Hybrid Search

Combines dense (vector) and sparse (BM25) search using Reciprocal Rank Fusion (RRF).

```rust
let config = RetrievalConfig {
    top_k: 10,
    alpha: 0.7,  // 70% dense, 30% sparse
    fusion_strategy: FusionStrategy::RRF { k: 60 },
    ..Default::default()
};
```

### RAPTOR Trees

Hierarchical clustering for improved long-document retrieval.

- **20%+ improvement** on QuALITY dataset
- **15%+ improvement** on NarrativeQA
- **18%+ improvement** on Qasper

```bash
rk-core query "explain the full methodology" --raptor
```

**See:** [API Reference - Retrieval System](API_REFERENCE.md#retrieval-system)

---

## Development

### Setup

```bash
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core
cargo build
```

### Run Tests

```bash
cargo test
cargo test --all-features
```

### Linting

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

### Documentation

```bash
# Generate rustdoc
cargo doc --no-deps --document-private-items

# Open in browser
cargo doc --no-deps --open
```

### Benchmarks

```bash
cargo bench
```

**See:** [ARCHITECTURE.md](../ARCHITECTURE.md)

---

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
cargo test --test '*'
```

### Benchmarks

```bash
cargo bench
```

**Expected Performance:**
- Embed single text (API): ~150ms
- Dense search (1M vectors): ~25ms
- BM25 search (1M docs): ~15ms
- Hybrid search (RRF): ~45ms

---

## Quality Gates

All code must pass 5 quality gates:

1. **Build**: `cargo build --release`
2. **Lint**: `cargo clippy -- -D warnings`
3. **Format**: `cargo fmt --check`
4. **Test**: `cargo test --all-features`
5. **Bench**: `cargo bench` (no >5% regression)

### Run All Gates

```bash
./scripts/quality_metrics.sh
```

**See:** [ORCHESTRATOR.md - Quality Gates](../ORCHESTRATOR.md#quality-gates-mandatory---cons-009)

---

## Rustdoc (Generated API Docs)

Auto-generated Rust documentation:

```bash
# Generate
cargo doc --no-deps --document-private-items

# Open in browser
cargo doc --no-deps --open
```

**Location:** `reasonkit-core/target/doc/reasonkit_core/index.html`

**Modules:**
- `reasonkit_core` - Core library
- `reasonkit_core::embedding` - Embedding pipeline
- `reasonkit_core::retrieval` - Retrieval system
- `reasonkit_core::storage` - Storage backends
- `reasonkit_core::thinktool` - ThinkTool protocol engine
- `reasonkit_core::raptor` - RAPTOR trees

---

## Examples

### RAG Pipeline

```rust
use reasonkit_core::rag::RagEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = RagEngine::new("./data").await?;

    let results = engine.query(
        "What is chain-of-thought reasoning?",
        5
    ).await?;

    for result in results {
        println!("{:.3} - {}", result.score, result.chunk.text);
    }

    Ok(())
}
```

### ThinkTool Execution

```rust
use reasonkit_core::thinktool::{ProtocolExecutor, ProtocolInput};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let executor = ProtocolExecutor::new()?;

    let result = executor.execute(
        "gigathink",
        ProtocolInput::query("Should we adopt microservices?")
    ).await?;

    println!("Confidence: {:.2}%", result.confidence * 100.0);

    Ok(())
}
```

**See:** [API Reference - Examples](API_REFERENCE.md#examples)

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Qdrant connection failed | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| OpenAI rate limit | Reduce batch size: `export REASONKIT_EMBEDDING_BATCH_SIZE="20"` |
| Out of memory | Process in smaller batches |
| Protocol timeout | Increase budget: `--budget "5m"` |

**See:** [CLI Reference - Troubleshooting](CLI_REFERENCE.md#troubleshooting)

---

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run quality gates: `./scripts/quality_metrics.sh`
5. Submit pull request

### Code Standards

- **Rust-first**: All production code in Rust
- **No `unsafe`**: Zero unsafe code
- **Test coverage**: >80% required
- **Documentation**: All public APIs documented
- **Quality gates**: All 5 gates must pass

---

## Resources

### Official

- **Website**: https://reasonkit.sh
- **Repository**: https://github.com/reasonkit/reasonkit-core
- **Documentation**: https://docs.rs/reasonkit-core
- **Discord**: https://discord.gg/reasonkit

### Papers

- **RAPTOR**: Sarthi et al., 2024 - Recursive Abstractive Processing
- **CoT**: Wei et al., 2022 - Chain-of-Thought Prompting
- **Self-Consistency**: Wang et al., 2022 - Self-Consistency Improves CoT
- **BGE-M3**: 2024 - Multilingual Embeddings

---

## License

Apache 2.0

---

## Version History

### v0.1.0 (2025-12-23)

Initial release:
- 5-layer RAG architecture
- Hybrid search (BM25 + vector)
- RAPTOR hierarchical retrieval
- 18+ LLM provider integrations
- 5 open-source ThinkTools
- Qdrant + in-memory + JSONL storage
- Complete CLI interface
- Comprehensive API documentation

---

**Last Updated:** 2025-12-23
**Maintainer:** ReasonKit Team
**Status:** Production Ready
