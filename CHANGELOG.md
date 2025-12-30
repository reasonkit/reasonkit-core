# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-12-30

### Added

- **Performance Optimizations (ThinkTool Executor)**
  - HTTP connection pooling via `reqwest::Client` reuse
  - Static regex caching with `once_cell::sync::Lazy`
  - Parallel step execution with `tokio::task::JoinSet`
  - Pre-allocation with `Vec::with_capacity()` for large collections

- **Agent Infrastructure**
  - GitHub agent configurations in `.github/agents/`
  - DevOps commands in `.claude/commands/devops/`
  - Agent handoff protocol documentation

- **Documentation**
  - BRAND_IDENTITY.md with complete design system
  - ROADMAP_2025.md with Q1 targets and milestones
  - Implementation workflow documentation
  - Architecture quick reference guide
  - Crates.io publication guide

### Changed

- **BREAKING**: Memory modules (embedding, storage, retrieval, raptor, indexing) extracted to `reasonkit-mem` crate
  - Use `reasonkit-core = { features = ["memory"] }` for automatic inclusion
  - Or install `reasonkit-mem` separately for standalone usage
- **Orchestration**: ORCHESTRATOR.md optimized from 56k to 12k chars (78% reduction)
- **Task Management**: Detailed docs moved to `~/TASKS/README.md` with prominent reference

### Removed

- `src/embedding/` - migrated to reasonkit-mem
- `src/storage/` - migrated to reasonkit-mem
- `src/retrieval/` - migrated to reasonkit-mem
- `src/raptor.rs` and `src/raptor_optimized.rs` - migrated to reasonkit-mem
- `src/indexing/` - migrated to reasonkit-mem
- Processed doc indices (`data/docs/processed/*.jsonl`)
- Model snapshots (`data/models/*.json`)

### Fixed

- Resolved potential memory leaks from HTTP client creation per request
- Fixed regex recompilation overhead in hot paths

### Migration

To migrate from v1.0.0:

```toml
# Option 1: Include memory with core (recommended)
[dependencies]
reasonkit-core = { version = "1.1", features = ["memory"] }

# Option 2: Use standalone memory crate
[dependencies]
reasonkit-core = "1.1"
reasonkit-mem = "0.1"
```

## [1.0.0] - 2025-12-25

### Added

- **TOML Protocol Support:** Support for defining custom ThinkTools in TOML files (`.toml`).
- **Benchmark Harness:** Unified benchmarking suite for ThinkTools, ingestion, and retrieval (`cargo bench --bench harness`).
- **Telemetry PII Stripping:** Privacy-first telemetry with regex-based PII redaction.
- **Web Research Command:** `rk-core web` command for deep research using DuckDuckGo/Tavily/Serper.
- **Verification Command:** `rk-core verify` command for 3-source claim triangulation.
- **MCP Integration:** Full Model Context Protocol support with `rk-core mcp` commands.
- **A/B Comparison Tool:** `rk-compare` binary for side-by-side raw vs ThinkTool-enhanced comparison.
- **Reasoning Profiles:** quick, balanced, deep, paranoid, and powercombo profiles.

### Changed

- **Documentation:** Major overhaul of API reference, troubleshooting, and security guides.
- **Project Structure:** Migrated RAG and Storage modules to `reasonkit-mem` (optional).
- **CLI:** Improved help output with clear descriptions and examples.

## [0.1.0] - 2025-12-25

### Added

- Initial release of `reasonkit-core`.
- **ThinkTools Engine:** Core implementation of GigaThink, LaserLogic, BedRock, ProofGuard, and BrutalHonesty.
- **CLI:** Basic `rk-core` command-line interface.
- **RAG:** Embedded Qdrant support and Tantivy integration.
- **LLM Clients:** Support for OpenAI, Anthropic, and OpenRouter.
- **PDF Ingestion:** Basic text extraction from PDFs.

### Fixed

- Fixed issue with Qdrant embedded mode on Linux.
- Resolved dependency conflicts in `Cargo.toml`.

[Unreleased]: https://github.com/reasonkit/reasonkit-core/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/reasonkit/reasonkit-core/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/reasonkit/reasonkit-core/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/reasonkit/reasonkit-core/releases/tag/v0.1.0
