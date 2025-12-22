# REASONKIT-CORE PROJECT CONTEXT
> Rust-first RAG/Knowledge Base Engine | Open Source (Apache 2.0)

**INHERITS FROM:** `../ORCHESTRATOR.md` (master orchestration document)
**LICENSE:** Apache 2.0 (fully open source)
**STATUS:** Primary OSS project - foundation for all ReasonKit products

---

## PROJECT OVERVIEW

```
reasonkit-core/
├── src/
│   ├── lib.rs          # Core types: Document, Chunk, SearchResult
│   ├── main.rs         # CLI: ingest, query, index, stats, serve
│   ├── error.rs        # Error handling (13 variants)
│   ├── ingestion/      # PDF, MD, HTML, JSON parsing
│   ├── embedding/      # OpenAI API + Local ONNX embeddings
│   ├── retrieval/      # Hybrid search (BM25 + vector)
│   ├── storage/        # Qdrant + local JSONL
│   ├── indexing/       # HNSW + Tantivy BM25
│   └── processing/     # Chunking, cleaning
├── data/
│   ├── papers/raw/     # 43 academic PDFs (101MB)
│   └── docs/           # 840 indexed documents
├── schemas/            # JSON schemas for documents
├── config/             # default.toml configuration
├── scripts/
│   ├── automation/     # Source validation, reindexing
│   └── download_*.sh   # Paper/doc downloaders
└── Cargo.toml          # Rust dependencies
```

---

## TECHNOLOGY STACK (PROJECT-SPECIFIC)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Vector DB | qdrant-client | 1.10+ | Dense embeddings storage |
| Full-text | tantivy | 0.22+ | BM25 lexical search |
| PDF Parser | lopdf | 0.33+ | Academic paper ingestion |
| Async Runtime | tokio | 1.x | Concurrent operations |
| CLI | clap | 4.x | Command-line interface |
| Parallelism | rayon | 1.10+ | Data-parallel processing |

### Build Commands

```bash
# Primary build
cargo build --release

# Run CLI
./target/release/rk-core --help
./target/release/rk-core ingest --path ./data/papers/raw --recursive
./target/release/rk-core query "chain of thought reasoning" --top-k 10

# Tests
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

---

## THINKTOOL ALLOCATION (OSS)

These ThinkTools are included in reasonkit-core (open source):

| Module | Shortcut | Purpose |
|--------|----------|---------|
| **GigaThink** | `gt` | Expansive creative thinking |
| **LaserLogic** | `ll` | Precision deductive reasoning |
| **BedRock** | `br` | First principles decomposition |
| **ProofGuard** | `pg` | Multi-source verification |
| **BrutalHonesty** | `bh` | Adversarial self-critique |

### Profile Mapping

| Profile | Modules | Confidence |
|---------|---------|------------|
| `--quick` | gt, ll | 70% |
| `--balanced` | gt, ll, br, pg | 80% |
| `--paranoid` | gt, ll, br, pg, bh | 95% |

---

## KNOWLEDGE BASE STATUS

### Academic Papers (43 PDFs, 101MB)

| Tier | Category | Count | Key Papers |
|------|----------|-------|------------|
| 1 | Revolutionary | 3 | DeepSeek-R1, COCONUT, AGoT |
| 2 | Advanced Reasoning | 5 | ToT, Self-Consistency, Reflexion |
| 3 | CoT Foundation | 2 | CoT Prompting, Zero-shot |
| 4-14 | Various | 33 | See `schemas/master_paper_sources.json` |

### Documentation Index (840 docs)

| Source | Documents |
|--------|-----------|
| Claude Code | 100 |
| Gemini CLI | 80 |
| Aider | 144 |
| Continue | 284 |
| MCP Protocol | 232 |

---

## DEVELOPMENT PRIORITIES

1. **RAPTOR Tree Structure** - Hierarchical retrieval (+20% on QuALITY)
2. **Qdrant Embedded Mode** - Zero-config local deployment
3. **Hybrid Search** - BM25 + vector with RRF fusion
4. **Reranking** - Cross-encoder for final precision

---

## TASKWARRIOR INTEGRATION

All work on this project MUST be tracked:

```bash
# Project structure
task project:rk-project.core.{component} "description" priority:{H|M|L}

# Components
# - rk-project.core.rag       → RAG implementation
# - rk-project.core.embedding → Embedding pipeline
# - rk-project.core.indexing  → Search indexes
# - rk-project.core.cli       → CLI improvements

# Start work
task {id} start  # Auto-starts timewarrior

# Complete work
task {id} done
```

---

## MCP SERVERS & SKILLS

When working on reasonkit-core, leverage:

- **MCP Sequential Thinking** - For complex reasoning chains
- **MCP Filesystem** - For data operations
- **pdf skill** - For PDF processing validation
- **math skill** - For benchmark calculations

---

## CONSTRAINTS (PROJECT-SPECIFIC)

| Constraint | Details |
|------------|---------|
| No Python in hot paths | All core logic in Rust |
| Test coverage > 80% | Required for merge |
| Benchmark regressions | Must not exceed 5% |
| Apache 2.0 compatible | All dependencies |

---

## PROTOCOLS

ThinkTool behavioral protocols are stored in `./protocols/`:

| Protocol | Shortcode | Profile | Description |
|----------|-----------|---------|-------------|
| ProofGuard Deep Research | `pg-deep` | paranoid | Multi-source triangulation with 5-phase workflow |

See `./protocols/README.md` for full protocol documentation.

---

## RELATED DOCUMENTS

- `../ORCHESTRATOR.md` - Master orchestration (inherited)
- `./ARCHITECTURE.md` - 5-layer RAG design
- `./docs/SOURCE_OVERVIEW.md` - All indexed sources
- `./protocols/` - ThinkTool behavioral protocols

---

*reasonkit-core v0.1.0 | Rust-first RAG | Apache 2.0*
