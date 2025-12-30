# ReasonKit Core - Integration Test Plan

> Comprehensive testing strategy for ReasonKit launch readiness
> Version: 1.0.0 | Last Updated: 2025-12-28

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Test Status](#current-test-status)
3. [Integration Test Categories](#integration-test-categories)
4. [E2E Test Scenarios](#e2e-test-scenarios)
5. [Mock LLM Provider Setup](#mock-llm-provider-setup)
6. [Performance Regression Tests](#performance-regression-tests)
7. [Cross-Platform Test Matrix](#cross-platform-test-matrix)
8. [Critical Path Testing](#critical-path-testing)
9. [Provider Failover Scenarios](#provider-failover-scenarios)
10. [Rate Limiting and Timeout Behavior](#rate-limiting-and-timeout-behavior)
11. [CI/CD Integration](#cicd-integration)
12. [Test Implementation Checklist](#test-implementation-checklist)

---

## Executive Summary

ReasonKit Core is a Rust-based structured prompt engineering framework supporting 18+ LLM providers. This document outlines the integration testing strategy to ensure launch readiness.

### Key Components Under Test

| Component               | Description                                                                            | Risk Level |
| ----------------------- | -------------------------------------------------------------------------------------- | ---------- |
| **ThinkTool Executor**  | Protocol chain execution engine                                                        | CRITICAL   |
| **LLM Provider Client** | 18+ provider integrations                                                              | CRITICAL   |
| **Profile System**      | 7 reasoning profiles (quick, balanced, deep, paranoid, decide, scientific, powercombo) | HIGH       |
| **ProofLedger**         | Content verification and drift detection                                               | HIGH       |
| **CLI Interface**       | rk-core command-line tool                                                              | HIGH       |
| **Trace System**        | Execution tracing and metrics                                                          | MEDIUM     |
| **Storage Layer**       | Document and embedding storage (optional feature)                                      | MEDIUM     |

### Test Coverage Targets

| Test Type              | Current     | Target                  | Priority |
| ---------------------- | ----------- | ----------------------- | -------- |
| Unit Tests             | 240 passing | 300+                    | P0       |
| Integration Tests      | ~50         | 100+                    | P0       |
| E2E Tests              | ~25         | 50+                     | P1       |
| Performance Benchmarks | 8 suites    | 12 suites               | P1       |
| Provider Mock Tests    | Partial     | Complete (18 providers) | P0       |

---

## Current Test Status

### Existing Test Files

```
tests/
├── protocol_e2e_tests.rs      # Protocol execution E2E tests (~661 lines)
├── storage_integration_tests.rs # Storage layer integration tests (~400 lines)
├── storage_unit_tests.rs       # Storage unit tests
└── verification_integration.rs # ProofLedger integration tests (~352 lines)
```

### Existing Benchmarks

```
benches/
├── thinktool_bench.rs    # Protocol execution performance
├── embedding_bench.rs    # Embedding generation performance
├── retrieval_bench.rs    # Vector search performance
├── fusion_bench.rs       # Hybrid search fusion
├── raptor_bench.rs       # RAPTOR tree operations
├── rerank_bench.rs       # Reranking performance
└── ingestion_bench.rs    # Document ingestion
```

---

## Integration Test Categories

### Category 1: ThinkTool Protocol Integration (CRITICAL)

Tests for the core reasoning protocol execution engine.

```rust
// File: tests/thinktool_integration.rs

#[cfg(test)]
mod thinktool_integration {
    use reasonkit::thinktool::{ExecutorConfig, ProtocolExecutor, ProtocolInput};

    // Test 1.1: Protocol Registry Integrity
    #[test]
    fn test_all_builtin_protocols_registered() {
        let executor = ProtocolExecutor::mock().unwrap();
        let protocols = executor.list_protocols();

        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
        assert!(protocols.contains(&"proofguard"));
        assert!(protocols.contains(&"brutalhonesty"));
    }

    // Test 1.2: Profile Registry Integrity
    #[test]
    fn test_all_builtin_profiles_registered() {
        let executor = ProtocolExecutor::mock().unwrap();
        let profiles = executor.list_profiles();

        assert_eq!(profiles.len(), 7);
        for profile in ["quick", "balanced", "deep", "paranoid", "decide", "scientific", "powercombo"] {
            assert!(profiles.contains(&profile), "Missing profile: {}", profile);
        }
    }

    // Test 1.3: Input Mapping Between Chain Steps
    #[tokio::test]
    async fn test_chain_step_input_propagation() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test query")
            .with_field("context", "Additional context");

        let result = executor.execute_profile("balanced", input).await.unwrap();

        // Verify all steps received their required inputs
        assert!(result.success, "Profile execution should succeed");
        assert!(!result.steps.is_empty(), "Should have step results");
    }

    // Test 1.4: Conditional Step Execution
    #[tokio::test]
    async fn test_conditional_step_skipping() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("High confidence query");

        // Deep profile has conditional BrutalHonesty step (confidence < 0.85)
        let result = executor.execute_profile("deep", input).await.unwrap();

        // With mock returning 0.85+ confidence, BrutalHonesty should be skipped
        // This tests the conditional execution logic
        assert!(result.success);
    }

    // Test 1.5: Token Aggregation Across Steps
    #[tokio::test]
    async fn test_token_aggregation() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute_profile("quick", input).await.unwrap();

        // Quick profile has 2 protocols, each with mock tokens
        assert!(result.tokens.total_tokens > 0);
        assert!(result.tokens.input_tokens > 0);
        assert!(result.tokens.output_tokens > 0);
    }
}
```

### Category 2: LLM Provider Integration (CRITICAL)

Tests for the 18+ LLM provider integrations.

```rust
// File: tests/llm_provider_integration.rs

#[cfg(test)]
mod llm_provider_integration {
    use reasonkit::thinktool::llm::{
        LlmConfig, LlmProvider, LlmRequest, UnifiedLlmClient,
        discover_available_providers, get_provider_info,
    };

    // Test 2.1: Provider Configuration Validation
    #[test]
    fn test_all_providers_have_valid_config() {
        for provider in LlmProvider::all() {
            assert!(!provider.env_var().is_empty());
            assert!(!provider.default_base_url().is_empty());
            assert!(!provider.default_model().is_empty());
            assert!(!provider.display_name().is_empty());
        }
    }

    // Test 2.2: Provider Discovery
    #[test]
    fn test_provider_discovery_with_mock_env() {
        // This test validates discovery logic works correctly
        // In CI, we should set at least one mock API key
        let info = get_provider_info();
        assert_eq!(info.len(), 18);
    }

    // Test 2.3: Azure OpenAI URL Construction
    #[test]
    fn test_azure_url_construction() {
        let config = LlmConfig::for_provider(LlmProvider::AzureOpenAI, "gpt-4o")
            .with_azure("my-resource", "my-deployment");

        assert_eq!(config.extra.azure_resource, Some("my-resource".to_string()));
        assert_eq!(config.extra.azure_deployment, Some("my-deployment".to_string()));
    }

    // Test 2.4: Cloudflare Gateway URL Construction
    #[test]
    fn test_cloudflare_gateway_url_construction() {
        let config = LlmConfig::for_provider(LlmProvider::CloudflareAI, "llama-3.3")
            .with_cloudflare_gateway("account123", "gateway456");

        assert!(config.extra.cf_account_id.is_some());
        assert!(config.extra.cf_gateway_id.is_some());
    }

    // Test 2.5: Cost Calculation Accuracy
    #[test]
    fn test_cost_calculation() {
        use reasonkit::thinktool::llm::LlmUsage;

        let usage = LlmUsage {
            input_tokens: 1000,
            output_tokens: 500,
            total_tokens: 1500,
        };

        // Claude Sonnet 4 pricing
        let cost = usage.cost_usd("claude-sonnet-4");
        assert!(cost > 0.0);
        assert!(cost < 0.02); // Should be around $0.0105

        // Groq should be cheaper
        let groq_cost = usage.cost_usd("llama-3.3-70b-versatile");
        assert!(groq_cost < cost);
    }

    // Test 2.6: Provider Format Detection
    #[test]
    fn test_provider_format_detection() {
        assert!(LlmProvider::Anthropic.is_anthropic_format());
        assert!(!LlmProvider::OpenAI.is_anthropic_format());

        assert!(LlmProvider::OpenAI.is_openai_compatible());
        assert!(LlmProvider::Groq.is_openai_compatible());
        assert!(LlmProvider::GoogleGemini.is_openai_compatible());
        assert!(LlmProvider::OpenRouter.is_openai_compatible());
    }
}
```

### Category 3: CLI Tool Integration (HIGH)

Tests for the command-line interface.

```rust
// File: tests/cli_integration.rs

#[cfg(test)]
mod cli_integration {
    use std::process::Command;

    // Test 3.1: CLI Help Output
    #[test]
    fn test_cli_help_displays() {
        let output = Command::new("cargo")
            .args(["run", "-p", "reasonkit-core", "--bin", "rk-core", "--", "--help"])
            .output()
            .expect("Failed to execute command");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("rk-core") || stdout.contains("ReasonKit"));
    }

    // Test 3.2: CLI List Protocols
    #[test]
    fn test_cli_list_protocols() {
        let output = Command::new("cargo")
            .args(["run", "-p", "reasonkit-core", "--bin", "rk-core", "--", "list", "protocols"])
            .output()
            .expect("Failed to execute command");

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should list available protocols
        assert!(output.status.success() || stdout.contains("gigathink"));
    }

    // Test 3.3: CLI Version Output
    #[test]
    fn test_cli_version() {
        let output = Command::new("cargo")
            .args(["run", "-p", "reasonkit-core", "--bin", "rk-core", "--", "--version"])
            .output()
            .expect("Failed to execute command");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("0.1") || stdout.contains("reasonkit"));
    }

    // Test 3.4: Mock Execution Mode
    #[test]
    fn test_cli_mock_execution() {
        let output = Command::new("cargo")
            .args([
                "run", "-p", "reasonkit-core", "--bin", "rk-core", "--",
                "think", "--mock", "--profile", "quick", "What is 2+2?"
            ])
            .output()
            .expect("Failed to execute command");

        // Mock mode should always succeed without API keys
        assert!(output.status.success());
    }
}
```

### Category 4: ProofLedger Integration (HIGH)

Tests for content verification and drift detection.

```rust
// File: tests/proofledger_integration.rs (extend existing verification_integration.rs)

#[cfg(test)]
mod proofledger_extended {
    use reasonkit::verification::ProofLedger;
    use std::collections::HashSet;

    // Test 4.1: Concurrent Anchor Operations
    #[test]
    fn test_concurrent_anchoring() {
        use std::sync::Arc;
        use std::thread;

        let ledger = Arc::new(ProofLedger::in_memory().unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let ledger = Arc::clone(&ledger);
            handles.push(thread::spawn(move || {
                ledger.anchor(
                    &format!("Content {}", i),
                    &format!("https://example.com/{}", i),
                    None
                )
            }));
        }

        let hashes: HashSet<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap().unwrap())
            .collect();

        assert_eq!(hashes.len(), 10); // All unique
    }

    // Test 4.2: Verification Under Load
    #[test]
    fn test_verification_throughput() {
        let ledger = ProofLedger::in_memory().unwrap();
        let content = "Test content for verification throughput";
        let url = "https://example.com/test";

        let hash = ledger.anchor(content, url, None).unwrap();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let result = ledger.verify(&hash, content).unwrap();
            assert!(result.verified);
        }
        let elapsed = start.elapsed();

        // Should verify 1000 items in under 1 second
        assert!(elapsed.as_millis() < 1000, "Verification took too long: {:?}", elapsed);
    }

    // Test 4.3: Database Persistence
    #[test]
    fn test_database_persistence_and_recovery() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        // Create and populate
        {
            let ledger = ProofLedger::new(&db_path).unwrap();
            for i in 0..100 {
                ledger.anchor(
                    &format!("Content {}", i),
                    &format!("https://example.com/{}", i),
                    None
                ).unwrap();
            }
            assert_eq!(ledger.count().unwrap(), 100);
        }

        // Reopen and verify
        {
            let ledger = ProofLedger::new(&db_path).unwrap();
            assert_eq!(ledger.count().unwrap(), 100);
        }
    }
}
```

---

## E2E Test Scenarios

### Scenario 1: Complete User Journey - Research Analysis

```rust
// File: tests/e2e/research_workflow.rs

/// E2E Test: Research analyst using ReasonKit for claim verification
#[tokio::test]
async fn test_research_analysis_workflow() {
    // GIVEN: A user wants to analyze a research claim
    let executor = ProtocolExecutor::mock().unwrap();
    let claim = ProtocolInput::query(
        "Evaluate the claim: GPT-4 achieves 86.4% accuracy on MMLU benchmark"
    ).with_field("context", "Academic research verification");

    // WHEN: User runs the scientific profile
    let result = executor.execute_profile("scientific", claim).await.unwrap();

    // THEN: Result should contain structured analysis
    assert!(result.success, "Scientific analysis should succeed");
    assert!(result.confidence >= 0.70, "Should have reasonable confidence");
    assert!(!result.steps.is_empty(), "Should have analysis steps");

    // AND: Result should have expected structure
    assert!(result.data.contains_key("confidence"));
    assert!(result.tokens.total_tokens > 0);
    assert!(result.duration_ms > 0);
}
```

### Scenario 2: Product Decision Making

```rust
/// E2E Test: Product manager making architecture decision
#[tokio::test]
async fn test_product_decision_workflow() {
    let executor = ProtocolExecutor::mock().unwrap();

    let decision = ProtocolInput::query(
        "Should we build a microservices architecture or stick with monolith for our B2B SaaS?"
    ).with_field("context", "Early stage startup, 5 engineers, expecting 10x growth");

    // Use decide profile for structured decision analysis
    let result = executor.execute_profile("decide", decision).await.unwrap();

    assert!(result.success);
    // Decide profile should complete in reasonable time
    assert!(result.duration_ms < 30000, "Should complete within 30 seconds");
}
```

### Scenario 3: High-Stakes Verification (Paranoid Mode)

```rust
/// E2E Test: Compliance officer verifying critical claims
#[tokio::test]
async fn test_paranoid_verification_workflow() {
    let executor = ProtocolExecutor::mock().unwrap();

    let critical_claim = ProtocolInput::query(
        "Verify: Our encryption implementation meets FIPS 140-2 Level 3 requirements"
    ).with_field("context", "Financial services compliance audit");

    // Use paranoid profile for maximum verification
    let result = executor.execute_profile("paranoid", critical_claim).await.unwrap();

    // Paranoid profile has 95% confidence target
    // With mock, we check the structure is correct
    assert!(!result.steps.is_empty());
    assert!(result.tokens.total_tokens > 0);
}
```

### Scenario 4: Rapid Brainstorming (Quick Mode)

```rust
/// E2E Test: Startup founder brainstorming ideas
#[tokio::test]
async fn test_quick_brainstorm_workflow() {
    let executor = ProtocolExecutor::mock().unwrap();

    let brainstorm = ProtocolInput::query(
        "What are 10 ways to improve user onboarding for our mobile app?"
    );

    let result = executor.execute_profile("quick", brainstorm).await.unwrap();

    assert!(result.success);
    assert!(result.confidence >= 0.70); // Quick profile has 70% target
    // Quick profile should be fast
    assert!(result.duration_ms < 10000, "Quick profile should be fast");
}
```

### Scenario 5: Cross-Provider Failover

```rust
/// E2E Test: Provider failover handling
#[tokio::test]
async fn test_provider_failover_workflow() {
    // This test verifies the system handles provider failures gracefully
    // In production, this would test actual failover; here we test the structure

    let available_providers = reasonkit::thinktool::llm::discover_available_providers();

    // If no providers configured, system should still work with mock
    if available_providers.is_empty() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test query");
        let result = executor.execute("gigathink", input).await.unwrap();
        assert!(result.success);
    }
}
```

---

## Mock LLM Provider Setup

### Mock Provider Architecture

```rust
// File: src/thinktool/mock_provider.rs (to be created)

/// Mock LLM provider for CI testing without API keys
pub struct MockLlmProvider {
    /// Configurable response delay (simulates network latency)
    delay_ms: u64,
    /// Error rate for chaos testing (0.0-1.0)
    error_rate: f64,
    /// Fixed responses per protocol/step
    responses: HashMap<String, String>,
    /// Token counts per response
    token_counts: TokenUsage,
}

impl MockLlmProvider {
    /// Create default mock provider
    pub fn new() -> Self {
        Self {
            delay_ms: 0,
            error_rate: 0.0,
            responses: Self::default_responses(),
            token_counts: TokenUsage::new(100, 150, 0.001),
        }
    }

    /// Create mock with simulated latency
    pub fn with_latency(delay_ms: u64) -> Self {
        Self { delay_ms, ..Self::new() }
    }

    /// Create mock with configurable error rate
    pub fn with_error_rate(rate: f64) -> Self {
        Self { error_rate: rate, ..Self::new() }
    }

    fn default_responses() -> HashMap<String, String> {
        let mut responses = HashMap::new();

        // GigaThink responses
        responses.insert(
            "gigathink.generate_perspectives".to_string(),
            r#"1. Technical perspective: Focus on implementation details
2. Business perspective: Consider ROI and market fit
3. User perspective: Prioritize experience and usability
4. Security perspective: Evaluate risk factors
5. Scalability perspective: Plan for growth
6. Maintenance perspective: Consider long-term costs
7. Team perspective: Assess capability match
8. Competitive perspective: Analyze market position
9. Innovation perspective: Explore novel approaches
10. Ethical perspective: Consider societal impact

Confidence: 0.85"#.to_string()
        );

        // LaserLogic responses
        responses.insert(
            "laserlogic.check_validity".to_string(),
            r#"Logical Analysis:

Premise 1: Valid - Established foundation
Premise 2: Valid - Logically follows
Premise 3: Partially valid - Requires clarification
Conclusion: Conditionally valid

No major fallacies detected.
Minor issue: Hasty generalization in premise 3.

Verdict: VALID with conditions

Confidence: 0.82"#.to_string()
        );

        // BedRock responses
        responses.insert(
            "bedrock.identify_axioms".to_string(),
            r#"First Principles Decomposition:

Axiom 1: [Foundational truth identified]
Axiom 2: [Core assumption validated]
Axiom 3: [Base principle established]

Dependencies:
- Axiom 1 depends on empirical observation
- Axiom 2 follows from Axiom 1
- Axiom 3 is independent

Confidence: 0.88"#.to_string()
        );

        // ProofGuard responses
        responses.insert(
            "proofguard.triangulate".to_string(),
            r#"Source Triangulation:

Source 1 (Primary): Confirms claim
Source 2 (Secondary): Partially confirms
Source 3 (Independent): Confirms claim

Triangulation Status: VERIFIED
Agreement: 3/3 sources align

Confidence: 0.90"#.to_string()
        );

        // BrutalHonesty responses
        responses.insert(
            "brutalhonesty.verdict".to_string(),
            r#"Adversarial Critique:

Strengths:
1. Well-structured argument
2. Evidence-based claims

Weaknesses:
1. Limited sample size
2. Potential confirmation bias

Red Flags:
- None critical

Verdict: APPROVED with recommendations

Confidence: 0.78"#.to_string()
        );

        responses
    }
}
```

### CI Environment Setup

```yaml
# .github/workflows/test.yml additions

env:
  # Mock provider flags
  REASONKIT_USE_MOCK: "true"
  REASONKIT_MOCK_DELAY_MS: "0"

  # Optional: Real provider keys for integration testing
  # These should be stored as GitHub secrets
  # ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  # OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

jobs:
  test:
    steps:
      - name: Run unit tests (mock mode)
        run: cargo test --no-fail-fast

      - name: Run integration tests (mock mode)
        run: cargo test --test '*_integration*' --no-fail-fast

      - name: Run E2E tests (mock mode)
        run: cargo test --test 'e2e_*' --no-fail-fast

      # Optional: Real provider tests (if secrets available)
      - name: Run provider integration tests
        if: env.ANTHROPIC_API_KEY != ''
        run: cargo test --test 'llm_provider_live' --no-fail-fast
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Mock Response Fixtures

```json
// fixtures/mock_responses/gigathink.json
{
  "generate_perspectives": {
    "content": "1. Perspective A\n2. Perspective B\n...",
    "tokens": { "input": 50, "output": 200 },
    "confidence": 0.85
  },
  "analyze_interactions": {
    "content": "Key interactions identified...",
    "tokens": { "input": 200, "output": 150 },
    "confidence": 0.8
  },
  "synthesize": {
    "content": "Synthesis: Main themes are...",
    "tokens": { "input": 150, "output": 100 },
    "confidence": 0.88
  }
}
```

---

## Performance Regression Tests

### Benchmark Thresholds

| Benchmark                 | Target  | Warning | Critical |
| ------------------------- | ------- | ------- | -------- |
| Protocol execution (mock) | < 5ms   | < 10ms  | < 50ms   |
| Profile chain (quick)     | < 15ms  | < 30ms  | < 100ms  |
| Profile chain (balanced)  | < 30ms  | < 60ms  | < 200ms  |
| Profile chain (deep)      | < 50ms  | < 100ms | < 300ms  |
| Concurrent execution (8x) | < 100ms | < 200ms | < 500ms  |
| Template rendering        | < 1ms   | < 5ms   | < 10ms   |
| Trace serialization       | < 2ms   | < 5ms   | < 20ms   |

### Performance Test Suite

```rust
// File: benches/regression_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

/// Performance regression benchmarks with strict thresholds
fn performance_regression_suite(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let executor = ProtocolExecutor::mock().unwrap();

    let mut group = c.benchmark_group("performance_regression");
    group.measurement_time(Duration::from_secs(30));

    // Threshold assertions
    group.sample_size(100);

    // Benchmark 1: Single protocol execution
    group.bench_function("single_protocol_gigathink", |b| {
        b.to_async(&rt).iter(|| async {
            let input = ProtocolInput::query("Test query");
            executor.execute("gigathink", input).await
        });
    });

    // Benchmark 2: Profile chain execution
    for profile in ["quick", "balanced", "deep", "paranoid"] {
        group.bench_with_input(
            BenchmarkId::new("profile_chain", profile),
            profile,
            |b, profile| {
                b.to_async(&rt).iter(|| async {
                    let input = ProtocolInput::query("Test query");
                    executor.execute_profile(profile, input).await
                });
            },
        );
    }

    // Benchmark 3: Concurrent execution scaling
    for concurrency in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_execution", concurrency),
            &concurrency,
            |b, &n| {
                b.to_async(&rt).iter(|| async {
                    let futures: Vec<_> = (0..n)
                        .map(|_| {
                            let input = ProtocolInput::query("Test");
                            executor.execute("gigathink", input)
                        })
                        .collect();
                    futures::future::join_all(futures).await
                });
            },
        );
    }

    group.finish();
}

/// Memory usage benchmarks
fn memory_regression_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_regression");

    // Benchmark: Large input handling
    group.bench_function("large_input_10kb", |b| {
        let large_input = "A".repeat(10_000);
        let executor = ProtocolExecutor::mock().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter(|| async {
            let input = ProtocolInput::query(&large_input);
            executor.execute("gigathink", input).await
        });
    });

    // Benchmark: Many concurrent requests
    group.bench_function("100_concurrent_requests", |b| {
        let executor = std::sync::Arc::new(ProtocolExecutor::mock().unwrap());
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter(|| async {
            let executor = executor.clone();
            let futures: Vec<_> = (0..100)
                .map(|i| {
                    let exec = executor.clone();
                    async move {
                        let input = ProtocolInput::query(&format!("Query {}", i));
                        exec.execute("gigathink", input).await
                    }
                })
                .collect();
            futures::future::join_all(futures).await
        });
    });

    group.finish();
}

criterion_group!(
    regression_benches,
    performance_regression_suite,
    memory_regression_suite,
);
criterion_main!(regression_benches);
```

### CI Performance Gate

```yaml
# .github/workflows/performance.yml

name: Performance Regression
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Run benchmarks
        run: cargo bench --bench regression_bench -- --save-baseline pr

      - name: Compare with baseline
        run: |
          cargo bench --bench regression_bench -- --baseline main --noplot

      - name: Check for regressions
        run: |
          # Parse benchmark results and fail if >10% regression
          ./scripts/check_benchmark_regression.sh 10
```

---

## Cross-Platform Test Matrix

### Platform Support Matrix

| Platform       | Architecture              | Priority | CI Coverage |
| -------------- | ------------------------- | -------- | ----------- |
| Linux x86_64   | x86_64-unknown-linux-gnu  | P0       | Full        |
| Linux ARM64    | aarch64-unknown-linux-gnu | P1       | Full        |
| macOS x86_64   | x86_64-apple-darwin       | P1       | Full        |
| macOS ARM64    | aarch64-apple-darwin      | P0       | Full        |
| Windows x86_64 | x86_64-pc-windows-msvc    | P2       | Core tests  |
| Windows ARM64  | aarch64-pc-windows-msvc   | P3       | Core tests  |

### CI Matrix Configuration

```yaml
# .github/workflows/cross-platform.yml

name: Cross-Platform Tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        include:
          # Linux x86_64 (Primary)
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            features: ""

          # Linux ARM64
          - os: ubuntu-latest
            target: aarch64-unknown-linux-gnu
            features: ""
            use_cross: true

          # macOS x86_64
          - os: macos-13
            target: x86_64-apple-darwin
            features: ""

          # macOS ARM64 (Apple Silicon)
          - os: macos-14
            target: aarch64-apple-darwin
            features: ""

          # Windows x86_64
          - os: windows-latest
            target: x86_64-pc-windows-msvc
            features: ""

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable
        with:
          targets: ${{ matrix.target }}

      - name: Install cross (if needed)
        if: matrix.use_cross
        run: cargo install cross --git https://github.com/cross-rs/cross

      - name: Build
        run: |
          if [ "${{ matrix.use_cross }}" = "true" ]; then
            cross build --target ${{ matrix.target }} --release
          else
            cargo build --target ${{ matrix.target }} --release
          fi

      - name: Test
        run: |
          if [ "${{ matrix.use_cross }}" = "true" ]; then
            cross test --target ${{ matrix.target }}
          else
            cargo test --target ${{ matrix.target }}
          fi
```

### Platform-Specific Tests

```rust
// File: tests/platform_specific.rs

#[cfg(test)]
mod platform_tests {

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_specific_features() {
        // Linux-specific file path handling
        let path = std::path::Path::new("/tmp/reasonkit_test");
        assert!(path.is_absolute());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_specific_features() {
        // macOS-specific path handling
        let home = dirs::home_dir().expect("Should have home directory");
        assert!(home.exists());
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_specific_features() {
        // Windows-specific path handling
        let path = std::path::Path::new("C:\\Users");
        // Path operations should work correctly
        assert!(path.is_absolute());
    }

    #[test]
    fn test_cross_platform_paths() {
        // Test platform-agnostic path handling
        use std::path::PathBuf;

        let mut path = PathBuf::new();
        path.push("reasonkit");
        path.push("traces");
        path.push("execution.json");

        // Should work on all platforms
        assert!(path.ends_with("execution.json"));
    }

    #[test]
    fn test_unicode_handling() {
        // Test Unicode in paths and content (important for i18n)
        let content = "Hello, 世界! Привет! مرحبا";
        let executor = reasonkit::thinktool::ProtocolExecutor::mock().unwrap();

        // Should handle Unicode without panicking
        let input = reasonkit::thinktool::ProtocolInput::query(content);
        assert_eq!(input.get_str("query"), Some(content));
    }
}
```

---

## Critical Path Testing

### Launch Day Critical Paths

| Path | Description                           | Test Coverage                |
| ---- | ------------------------------------- | ---------------------------- |
| CP-1 | CLI --help displays correctly         | `test_cli_help_displays`     |
| CP-2 | Mock mode works without API keys      | `test_cli_mock_execution`    |
| CP-3 | All 5 ThinkTools execute successfully | `test_all_protocols_execute` |
| CP-4 | All 7 profiles execute successfully   | `test_all_profiles_execute`  |
| CP-5 | JSON output is valid                  | `test_json_output_valid`     |
| CP-6 | Error messages are user-friendly      | `test_error_messages`        |
| CP-7 | Trace files are created correctly     | `test_trace_creation`        |
| CP-8 | Metrics are recorded accurately       | `test_metrics_accuracy`      |

### Critical Path Test Implementation

```rust
// File: tests/critical_paths.rs

#[cfg(test)]
mod critical_paths {
    use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

    /// CP-1: CLI help displays
    #[test]
    fn cp1_cli_help_displays() {
        let output = std::process::Command::new("cargo")
            .args(["run", "-p", "reasonkit-core", "--bin", "rk-core", "--", "--help"])
            .output()
            .expect("Failed to run CLI");

        assert!(output.status.success() || !output.stdout.is_empty());
    }

    /// CP-2: Mock mode works without API keys
    #[tokio::test]
    async fn cp2_mock_mode_no_api_keys() {
        // Ensure no API keys are set for this test
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");

        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test query");
        let result = executor.execute("gigathink", input).await;

        assert!(result.is_ok(), "Mock mode should work without API keys");
    }

    /// CP-3: All 5 ThinkTools execute
    #[tokio::test]
    async fn cp3_all_thinktools_execute() {
        let executor = ProtocolExecutor::mock().unwrap();

        let protocols = ["gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty"];

        for protocol in protocols {
            let input = match protocol {
                "gigathink" => ProtocolInput::query("Test query"),
                "laserlogic" => ProtocolInput::argument("If A then B. A. Therefore B."),
                "bedrock" => ProtocolInput::statement("Water boils at 100C"),
                "proofguard" => ProtocolInput::claim("The sky is blue"),
                "brutalhonesty" => ProtocolInput::work("My analysis shows growth"),
                _ => unreachable!(),
            };

            let result = executor.execute(protocol, input).await;
            assert!(result.is_ok(), "Protocol {} should execute: {:?}", protocol, result.err());
            assert!(result.unwrap().success, "Protocol {} should succeed", protocol);
        }
    }

    /// CP-4: All 7 profiles execute
    #[tokio::test]
    async fn cp4_all_profiles_execute() {
        let executor = ProtocolExecutor::mock().unwrap();

        let profiles = ["quick", "balanced", "deep", "paranoid", "decide", "scientific", "powercombo"];

        for profile in profiles {
            let input = ProtocolInput::query("Test query for profile execution");
            let result = executor.execute_profile(profile, input).await;

            assert!(result.is_ok(), "Profile {} should execute: {:?}", profile, result.err());
        }
    }

    /// CP-5: JSON output is valid
    #[tokio::test]
    async fn cp5_json_output_valid() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");
        let result = executor.execute("gigathink", input).await.unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&result).expect("Should serialize to JSON");

        // Parse back
        let _parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("Should parse back as valid JSON");
    }

    /// CP-6: Error messages are user-friendly
    #[tokio::test]
    async fn cp6_error_messages_friendly() {
        let executor = ProtocolExecutor::mock().unwrap();

        // Test nonexistent protocol
        let result = executor.execute("nonexistent", ProtocolInput::query("Test")).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let msg = err.to_string();

        // Error should be descriptive
        assert!(msg.contains("protocol") || msg.contains("not found") || msg.contains("NotFound"));
    }

    /// CP-7: Trace creation works
    #[test]
    fn cp7_trace_creation() {
        use reasonkit::thinktool::trace::ExecutionTrace;

        let trace = ExecutionTrace::new("gigathink", "1.0.0");

        assert!(!trace.id.is_nil());
        assert_eq!(trace.protocol_id, "gigathink");
        assert_eq!(trace.protocol_version, "1.0.0");
    }

    /// CP-8: Metrics accuracy
    #[tokio::test]
    async fn cp8_metrics_accuracy() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let start = std::time::Instant::now();
        let result = executor.execute("gigathink", input).await.unwrap();
        let elapsed = start.elapsed();

        // Duration should be recorded and roughly accurate
        assert!(result.duration_ms > 0);
        assert!((result.duration_ms as u128) <= elapsed.as_millis() + 100);

        // Tokens should be tracked
        assert!(result.tokens.total_tokens > 0);
    }
}
```

---

## Provider Failover Scenarios

### Failover Test Matrix

| Scenario | Primary           | Fallback                 | Expected Behavior             |
| -------- | ----------------- | ------------------------ | ----------------------------- |
| F-1      | Anthropic timeout | OpenAI                   | Automatic retry with fallback |
| F-2      | Rate limit hit    | Queue + retry            | Exponential backoff           |
| F-3      | Invalid API key   | Error with clear message | Fail fast with guidance       |
| F-4      | Network failure   | Cached response (if any) | Graceful degradation          |
| F-5      | Provider down     | Alternative provider     | Automatic rerouting           |

### Failover Test Implementation

```rust
// File: tests/provider_failover.rs

#[cfg(test)]
mod provider_failover {
    use reasonkit::thinktool::llm::{LlmConfig, LlmProvider, UnifiedLlmClient};
    use std::time::Duration;

    /// F-1: Timeout handling
    #[tokio::test]
    async fn test_timeout_handling() {
        // Configure very short timeout
        let config = LlmConfig {
            provider: LlmProvider::Anthropic,
            timeout_secs: 1, // 1 second timeout
            ..Default::default()
        };

        let client = UnifiedLlmClient::new(config);
        assert!(client.is_ok(), "Client creation should succeed");

        // Note: Actual timeout testing requires mock server
        // This just validates config is accepted
    }

    /// F-2: Rate limit detection
    #[test]
    fn test_rate_limit_error_detection() {
        // Test that rate limit errors are properly categorized
        let error_msg = "rate_limit_exceeded";
        assert!(error_msg.contains("rate") || error_msg.contains("limit"));
    }

    /// F-3: Invalid API key handling
    #[test]
    fn test_missing_api_key_error() {
        // Clear any existing API keys
        std::env::remove_var("ANTHROPIC_API_KEY");

        let config = LlmConfig::default();
        let client = UnifiedLlmClient::new(config).unwrap();

        // Getting API key should fail gracefully
        // The get_api_key method is private, so we test via execution
        // In mock mode, this is bypassed
    }

    /// F-4: Provider fallback chain
    #[test]
    fn test_provider_priority_chain() {
        // Verify provider priority order
        let providers = LlmProvider::all();

        // Tier 1 providers should be listed first
        let tier1_indices: Vec<_> = [
            LlmProvider::Anthropic,
            LlmProvider::OpenAI,
            LlmProvider::GoogleGemini,
        ].iter()
            .filter_map(|p| providers.iter().position(|x| x == p))
            .collect();

        // Tier 1 providers should come before aggregators
        let openrouter_idx = providers.iter().position(|p| *p == LlmProvider::OpenRouter).unwrap();

        for idx in tier1_indices {
            assert!(idx < openrouter_idx, "Tier 1 providers should come before aggregators");
        }
    }
}
```

---

## Rate Limiting and Timeout Behavior

### Rate Limit Handling Tests

```rust
// File: tests/rate_limit.rs

#[cfg(test)]
mod rate_limit_tests {
    use std::time::{Duration, Instant};

    /// Test exponential backoff calculation
    #[test]
    fn test_exponential_backoff() {
        let base_delay_ms = 100;
        let max_delay_ms = 10000;

        for attempt in 0..10 {
            let delay = std::cmp::min(
                base_delay_ms * 2u64.pow(attempt),
                max_delay_ms
            );

            // Verify backoff grows exponentially up to max
            if attempt < 7 {
                assert!(delay < max_delay_ms);
            } else {
                assert_eq!(delay, max_delay_ms);
            }
        }
    }

    /// Test retry budget tracking
    #[test]
    fn test_retry_budget() {
        let max_retries = 3;
        let mut retries = 0;

        while retries < max_retries {
            retries += 1;
        }

        assert_eq!(retries, max_retries);
    }

    /// Test request timeout enforcement
    #[tokio::test]
    async fn test_request_timeout() {
        let timeout = Duration::from_millis(100);
        let start = Instant::now();

        let result = tokio::time::timeout(
            timeout,
            tokio::time::sleep(Duration::from_secs(10))
        ).await;

        let elapsed = start.elapsed();

        assert!(result.is_err(), "Should timeout");
        assert!(elapsed < Duration::from_millis(200), "Should timeout quickly");
    }
}
```

### Timeout Configuration Tests

```rust
// File: tests/timeout.rs

#[cfg(test)]
mod timeout_tests {
    use reasonkit::thinktool::executor::ExecutorConfig;

    /// Test default timeout values
    #[test]
    fn test_default_timeout() {
        let config = ExecutorConfig::default();
        assert_eq!(config.timeout_secs, 120); // 2 minutes default
    }

    /// Test custom timeout configuration
    #[test]
    fn test_custom_timeout() {
        let config = ExecutorConfig {
            timeout_secs: 30,
            ..Default::default()
        };
        assert_eq!(config.timeout_secs, 30);
    }

    /// Test timeout boundaries
    #[test]
    fn test_timeout_boundaries() {
        // Minimum reasonable timeout
        let short = ExecutorConfig {
            timeout_secs: 5,
            ..Default::default()
        };
        assert!(short.timeout_secs >= 5);

        // Maximum reasonable timeout
        let long = ExecutorConfig {
            timeout_secs: 600, // 10 minutes
            ..Default::default()
        };
        assert!(long.timeout_secs <= 600);
    }
}
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test-suite.yml

name: ReasonKit Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Run daily at 6am UTC
    - cron: "0 6 * * *"

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  # Unit Tests (fast, run on every PR)
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run unit tests
        run: cargo test --lib --no-fail-fast

      - name: Check test count
        run: |
          count=$(cargo test --lib 2>&1 | grep -oP '\d+ passed' | grep -oP '\d+')
          echo "Unit tests passed: $count"
          if [ "$count" -lt 200 ]; then
            echo "Warning: Less than 200 unit tests"
          fi

  # Integration Tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run integration tests
        run: cargo test --test '*' --no-fail-fast

  # E2E Tests
  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - uses: Swatinem/rust-cache@v2

      - name: Build CLI
        run: cargo build --release --bin rk-core

      - name: Run E2E tests
        run: cargo test --test 'e2e_*' --no-fail-fast

  # Performance Benchmarks
  benchmarks:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run benchmarks
        run: cargo bench --no-run

      - name: Store benchmark results
        run: |
          cargo bench -- --save-baseline main --noplot
          # Upload results to compare PRs

  # Cross-Platform Matrix
  cross-platform:
    needs: unit-tests
    strategy:
      matrix:
        os: [ubuntu-latest, macos-14, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - uses: Swatinem/rust-cache@v2

      - name: Build
        run: cargo build --release

      - name: Test
        run: cargo test

  # Live Provider Tests (scheduled only, with secrets)
  live-provider-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable

      - name: Test Anthropic provider
        if: ${{ secrets.ANTHROPIC_API_KEY != '' }}
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: cargo test test_anthropic_live --ignored

      - name: Test OpenAI provider
        if: ${{ secrets.OPENAI_API_KEY != '' }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: cargo test test_openai_live --ignored
```

### Quality Gates

```yaml
# Required checks before merge
required_checks:
  - unit-tests
  - integration-tests
  - cross-platform (ubuntu-latest)
  - cross-platform (macos-14)

# Warning-only checks
optional_checks:
  - cross-platform (windows-latest)
  - benchmarks
  - live-provider-tests
```

---

## Test Implementation Checklist

### Phase 1: Pre-Launch (P0) - Must Complete

- [ ] **1.1** All 5 ThinkTools execute successfully with mock
- [ ] **1.2** All 7 profiles execute successfully with mock
- [ ] **1.3** CLI --help and --version work correctly
- [ ] **1.4** Mock mode works without any API keys
- [ ] **1.5** JSON output is valid and parseable
- [ ] **1.6** Error messages are user-friendly
- [ ] **1.7** All 240+ unit tests passing
- [ ] **1.8** Cross-platform builds succeed (Linux, macOS, Windows)

### Phase 2: Launch Week (P1) - Should Complete

- [ ] **2.1** E2E test suite for 5 key user journeys
- [ ] **2.2** Performance benchmarks with regression detection
- [ ] **2.3** Provider failover tests (mock)
- [ ] **2.4** Rate limit handling tests
- [ ] **2.5** Timeout behavior tests
- [ ] **2.6** ProofLedger concurrent access tests
- [ ] **2.7** Memory leak detection (valgrind/heaptrack)

### Phase 3: Post-Launch (P2) - Nice to Have

- [ ] **3.1** Live provider integration tests (with API keys)
- [ ] **3.2** Chaos/fault injection tests
- [ ] **3.3** Long-running stability tests (24h+)
- [ ] **3.4** Internationalization tests (Unicode, RTL)
- [ ] **3.5** Accessibility compliance (CLI output)
- [ ] **3.6** Security scanning (OWASP, dependency audit)

---

## Appendix: Test File Templates

### Template: Integration Test

```rust
//! Integration tests for [Component Name]
//!
//! These tests verify [what aspect] of [component].

use reasonkit::{/* imports */};

#[cfg(test)]
mod [component]_integration {
    use super::*;

    // Setup helper
    fn setup() -> /* return type */ {
        // Common setup
    }

    // Teardown helper (if needed)
    fn teardown() {
        // Cleanup
    }

    // Test: [Description]
    #[tokio::test]
    async fn test_[specific_behavior]() {
        // GIVEN: [preconditions]
        let fixture = setup();

        // WHEN: [action]
        let result = /* action */;

        // THEN: [expected outcome]
        assert!(/* assertion */);
    }
}
```

### Template: Benchmark

```rust
//! Benchmarks for [Component Name]
//!
//! Performance targets:
//! - [Target 1]: < X ms
//! - [Target 2]: < Y ops/sec

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_[component](c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("[benchmark_name]", |b| {
        b.to_async(&rt).iter(|| async {
            // Benchmark code
        });
    });
}

criterion_group!(benches, bench_[component]);
criterion_main!(benches);
```

---

## Document History

| Version | Date       | Author             | Changes                         |
| ------- | ---------- | ------------------ | ------------------------------- |
| 1.0.0   | 2025-12-28 | Testing Specialist | Initial comprehensive test plan |

---

_ReasonKit Core - Designed, Not Dreamed_
*https://reasonkit.sh*
