# GLM-4.6 Integration Module

**Status:** Experimental (feature-gated) | **Version:** 0.1.0 | **Last Updated:** 2026-01-03

## Overview

Complete integration for GLM-4.6 (Zhipu AI) model in ReasonKit, providing:

- **198K Token Context Window**: Expanded via YaRN extension
- **Cost Efficiency**: 1/7th Claude pricing with performance tracking
- **Agentic Excellence**: 70.1% TAU-Bench score for coordination tasks
- **Structured Output**: Superior format adherence
- **Bilingual Support**: Chinese/English optimization

## Module Structure

```
glm46/
├── mod.rs                 # Module exports
├── client.rs              # Async HTTP client for GLM-4.6 API
├── types.rs               # Type definitions (ChatRequest, ChatResponse, etc.)
├── thinktool_profile.rs   # ThinkTool profile integration
├── circuit_breaker.rs     # Fault tolerance
├── mcp_server.rs          # MCP server for agent coordination
├── ollama.rs              # Local deployment support
└── orchestrator.rs        # Multi-agent orchestration
```

## Features

### 1. High-Performance Client (`client.rs`)

- Async/await with tokio
- Cost tracking and optimization
- Circuit breaker for fault tolerance
- Local fallback via ollama
- 198K context window support

### 2. ThinkTool Integration (`thinktool_profile.rs`)

- GLM-4.6 enhanced GigaThink
- GLM-4.6 enhanced LaserLogic
- Structured output mastery
- Bilingual optimization

### 3. MCP Server (`mcp_server.rs`)

- Agent coordination tools
- Workflow optimization
- Conflict resolution
- Multi-agent orchestration

### 4. Circuit Breaker (`circuit_breaker.rs`)

- Fault tolerance
- Graceful degradation
- Automatic recovery

## Usage

### Basic Client Usage

```rust
use reasonkit_core::glm46::{GLM46Client, GLM46Config};

let config = GLM46Config {
    api_key: std::env::var("GLM46_API_KEY")?,
    context_budget: 198_000,
    ..Default::default()
};

let client = GLM46Client::new(config)?;
```

### ThinkTool Profile Usage

```rust
use reasonkit_core::glm46::thinktool_profile::GLM46ThinkToolProfile;

let profile = GLM46ThinkToolProfile::new(config);
let result = profile.execute_reasoning_chain("query", "balanced").await?;
```

## Testing Infrastructure

Comprehensive test suite created:

- **Unit Tests** (`tests/glm46_unit_tests.rs`): 15+ tests for config, types, serialization
- **Integration Tests** (`tests/glm46_integration_tests.rs`): 8+ E2E tests
- **Benchmarks** (`benches/glm46_benchmark.rs`): 7 performance benchmarks
- **Performance Validation** (`tests/glm46_performance_validation.rs`): TAU-Bench, context window, cost efficiency validation

### Running Tests

```bash
# Unit tests (feature must be enabled)
cargo test --features glm46 --test glm46_unit_tests

# Integration tests (requires GLM46_API_KEY)
GLM46_API_KEY=your_key cargo test --features glm46 --test glm46_integration_tests

# Benchmarks
cargo bench --features glm46 --bench glm46_benchmark
```

## Performance Targets

| Metric           | Target       | Status                |
| ---------------- | ------------ | --------------------- |
| TAU-Bench Score  | 70.1%        | ⏳ Validation pending |
| Context Window   | 198K tokens  | ✅ Supported          |
| Cost Efficiency  | 1/7th Claude | ✅ Tracked            |
| Latency Overhead | <5ms         | ⏳ Benchmarking       |

## Current Status

### Compilation Status

**Current Status:** ✅ Compiles successfully (feature-gated)

This module is **experimental** and must be explicitly enabled:

```bash
cargo build --features glm46
cargo test --features glm46
```

**Completed:**

- ✅ Core module compilation: Fixed
- ✅ Error handling: Properly typed
- ✅ Feature gating: Module is opt-in
- ✅ Test infrastructure: Feature-gated

**Note:** This module requires the `glm46` feature flag. It is not included in default builds.

## Configuration

### Environment Variables

- `GLM46_API_KEY`: API key for GLM-4.6 (required)
- `GLM46_BASE_URL`: Base URL (default: `https://openrouter.ai/api/v1`)

### Default Configuration

```rust
GLM46Config {
    model: "glm-4.6",
    timeout: Duration::from_secs(30),
    context_budget: 198_000,
    cost_tracking: true,
    local_fallback: true,
}
```

## Cost Tracking

GLM-4.6 pricing (via OpenRouter, approximate):

- Input: $0.0001 per 1K tokens
- Output: $0.0002 per 1K tokens
- **Ratio vs Claude**: ~1/7th (Claude: $0.008/$0.024 per 1K)

Cost tracking is automatic when `cost_tracking: true` in config.

## Local Deployment

Supports local deployment via ollama:

```rust
let config = GLM46Config {
    base_url: "http://localhost:11434".to_string(),
    local_fallback: true,
    ..Default::default()
};
```

## Documentation

- **API Documentation**: `cargo doc --open --package reasonkit-core --features glm46`
- **Test Documentation**: See test files for usage examples

## Contributing

When contributing to GLM-4.6 integration:

1. Run tests: `cargo test --features glm46`
2. Check benchmarks: `cargo bench --features glm46 --bench glm46_benchmark`
3. Follow Rust best practices
4. Update this README with changes

## References

- [GLM-4.6 Documentation](https://open.bigmodel.cn/)
- [OpenRouter API](https://openrouter.ai/docs)
- [TAU-Bench](https://github.com/taubench/taubench) - Agentic coordination benchmark

---

**Status**: Experimental (feature-gated) ✅ | Enable with `--features glm46`
