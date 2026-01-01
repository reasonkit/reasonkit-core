# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-01

### Added

#### ThinkTools Engine (Core Feature)

- **GigaThink** (`gt`): Multi-perspective expansion generating 10+ viewpoints for comprehensive analysis
- **LaserLogic** (`ll`): Precision deductive reasoning with fallacy detection and gap analysis
- **BedRock** (`br`): First principles decomposition with axiom identification and rebuilding
- **ProofGuard** (`pg`): Multi-source triangulation requiring 3+ independent sources for verification
- **BrutalHonesty** (`bh`): Adversarial self-critique that actively challenges conclusions

#### Reasoning Profiles

- **quick**: Fast analysis using GigaThink + LaserLogic (70% confidence threshold)
- **balanced**: Standard analysis with 4 ThinkTools (80% confidence threshold)
- **deep**: Comprehensive analysis using all 5 ThinkTools (85% confidence threshold)
- **paranoid**: Maximum rigor with all ThinkTools + validation (95% confidence threshold)
- **powercombo**: Full chain execution for variance reduction

#### MiniMax M2 Integration (Experimental)

- M2 protocol types and service builder (`m2` module)
- Interleaved thinking engine for enhanced reasoning
- Task classification system with use case mapping
- Protocol generator for dynamic protocol creation
- Enhanced ThinkTool variants (feature-gated under `minimax`)

#### Validation Engine

- **DeepSeekValidationEngine**: Statistical and methodological validation
- Compliance checking with configurable violation thresholds
- Bias detection across multiple categories
- Regulatory status tracking for compliance workflows
- Integration with ValidatingProtocolExecutor for inline validation

#### LLM Provider Support (18+ Providers)

- **Major Cloud**: Anthropic, OpenAI, Google Gemini, Vertex AI, Azure OpenAI, AWS Bedrock
- **Specialized**: xAI (Grok), Groq, Mistral, DeepSeek, Cohere, Perplexity, Cerebras
- **Inference**: Together AI, Fireworks AI, Alibaba Qwen
- **Aggregation**: OpenRouter (300+ models), Cloudflare AI Gateway
- Provider auto-discovery via `discover_available_providers()`
- Unified client interface via `UnifiedLlmClient`

#### CLI Commands

- `rk-core think --profile <profile> "<query>"`: Execute reasoning with specified profile
- `rk-core mcp`: Model Context Protocol management
- `rk-core serve-mcp`: Start MCP server
- `rk-core completions`: Shell completion generation
- `rk-compare`: Side-by-side raw vs ThinkTool-enhanced comparison

#### Benchmarking Infrastructure

- `BenchmarkRunner` for systematic ThinkTool evaluation
- `CalibrationTracker` for confidence score calibration
- `MetricsTracker` for execution performance monitoring
- Criterion benchmarks for retrieval, fusion, embedding, and ingestion
- Variance reduction benchmarks with statistical analysis

#### Advanced Reasoning Modules

- **Tree of Thoughts (ToT)**: Branching exploration with backtracking
- **Socratic Dialogue**: Question-driven reasoning with aporia detection
- **Toulmin Argumentation**: Structured argument analysis with warrants and rebuttals
- **Formal Logic (FOL)**: First-order logic formalization and soundness checking
- **Debate Arena**: Multi-agent adversarial reasoning
- **Self-Consistency**: Multiple reasoning path aggregation
- **Process Reward Model (PRM)**: Step-level scoring and reranking

#### Telemetry and Tracing

- SQLite-backed execution traces for audit and replay
- PII stripping with regex-based redaction
- Privacy-first telemetry with configurable retention
- Execution status tracking per step

#### Optional Features

- `memory`: Integration with `reasonkit-mem` for storage, retrieval, and embeddings
- `local-embeddings`: BGE-M3 ONNX local inference via `ort` and `tokenizers`
- `aesthetic`: M2-enhanced UI/UX assessment system
- `vibe`: VIBE Protocol validation system (experimental)
- `code-intelligence`: Multi-language code intelligence (experimental)
- `arf`: Autonomous Reasoning Framework module

### Changed

- **BREAKING**: Memory modules (storage, embedding, retrieval, raptor, indexing) extracted to `reasonkit-mem` crate
- **BREAKING**: RAG module now requires `memory` feature flag
- Project structure reorganized with clear separation between core reasoning and memory infrastructure
- CLI help output improved with detailed descriptions and usage examples
- Documentation overhauled with API reference, troubleshooting, and security guides

### Fixed

- Invalid syntax in `validation.rs` string literals
- M2 protocol type definitions finalized for stable API
- Qdrant embedded mode compatibility on Linux
- Dependency conflicts in `Cargo.toml` resolved
- SVG image rendering issues in README (replaced with PNG alternatives)

### Security

- Pre-launch security cleanup removing sensitive files from tracking
- Codebase hardening with security audit
- No hardcoded secrets (environment variable configuration only)
- GDPR-compliant telemetry with PII redaction

### Performance

- Target orchestration latency: <10ms (measured ~7ms)
- Variance reduction from ~85% (raw LLM) to ~28% (after ThinkTool chain)
- Parallel execution support via `rayon` for batch operations
- Async runtime via `tokio` with full feature set

## [0.2.0] - 2025-12-30

### Added

- TOML protocol support for custom ThinkTool definitions
- Unified benchmarking harness (`cargo bench --bench harness`)
- Telemetry with PII stripping and privacy controls
- Web research command (`rk-core web`) for DuckDuckGo/Tavily/Serper integration
- Verification command (`rk-core verify`) for 3-source claim triangulation
- Full MCP integration with `rk-core mcp` commands
- A/B comparison tool (`rk-compare` binary)

### Changed

- Major documentation overhaul
- Project structure migration (RAG/Storage to `reasonkit-mem`)
- CLI improvements with clearer help output

## [0.1.0] - 2025-12-25

### Added

- Initial release of `reasonkit-core`
- Core ThinkTools implementation (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty)
- Basic CLI (`rk-core`)
- Embedded Qdrant and Tantivy integration for RAG
- LLM clients for OpenAI, Anthropic, and OpenRouter
- Basic PDF text extraction

### Fixed

- Qdrant embedded mode on Linux
- Dependency conflicts in `Cargo.toml`

---

## Migration Guide

### From 0.2.x to 0.3.0

1. **Memory Feature Flag**: If you use storage, retrieval, or embedding functionality, add the `memory` feature:
   ```toml
   [dependencies]
   reasonkit-core = { version = "0.3", features = ["memory"] }
   ```

2. **Import Changes**: Memory types are now re-exported from `reasonkit-mem`:
   ```rust
   // Old
   use reasonkit::storage::DocumentStore;

   // New (with memory feature)
   use reasonkit::storage::DocumentStore;  // Re-exported from reasonkit-mem
   ```

3. **Validation Engine**: New validation capabilities are available via `thinktool::validation`:
   ```rust
   use reasonkit::thinktool::{DeepSeekValidationEngine, DeepSeekValidationConfig};
   ```

### From 0.1.x to 0.2.x

No breaking changes. TOML protocol files are now supported alongside YAML.

---

## Links

[Unreleased]: https://github.com/reasonkit/reasonkit-core/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/reasonkit/reasonkit-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/reasonkit/reasonkit-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/reasonkit/reasonkit-core/releases/tag/v0.1.0
