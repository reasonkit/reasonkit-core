# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-01-01

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

[Unreleased]: https://github.com/reasonkit/reasonkit-core/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/reasonkit/reasonkit-core/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/reasonkit/reasonkit-core/releases/tag/v0.1.0
