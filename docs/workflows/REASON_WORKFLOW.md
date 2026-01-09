# ReasonKit Core: Query-Reason-Response Workflow

## Executive Summary

This document traces the complete Query -> Reason -> Response workflow through the ReasonKit Core codebase, documenting how queries are processed, ThinkTools are selected and executed, traces are captured, and responses are assembled.

**Status**: Verified as of 2026-01-01

---

## 1. Workflow Architecture Overview

```
                            +-------------------+
                            |   CLI / API       |
                            |   (main.rs)       |
                            +--------+----------+
                                     |
                                     v
+-------------------+    +------------------------+    +-------------------+
|   ProtocolInput   |--->|   ProtocolExecutor     |--->|  ProtocolOutput   |
|   - query         |    |   (executor.rs)        |    |  - success        |
|   - fields        |    |   - registry           |    |  - confidence     |
|   - context       |    |   - profiles           |    |  - steps[]        |
+-------------------+    |   - llm_client         |    |  - tokens         |
                         +------------------------+    |  - trace_id       |
                                     |                 +-------------------+
                    +----------------+----------------+
                    |                |                |
                    v                v                v
           +--------+-----+  +-------+------+  +-----+-------+
           | Protocol     |  | LLM Client   |  | Trace       |
           | Registry     |  | (llm.rs)     |  | (trace.rs)  |
           | (registry.rs)|  +--------------+  +-------------+
           +--------------+
                    |
                    v
           +--------+-----+
           | ThinkTool    |
           | Modules      |
           | (modules/)   |
           +--------------+
```

---

## 2. Component-by-Component Trace

### 2.1 Entry Point: CLI (main.rs)

**Location**: `src/main.rs`

The CLI parses commands and routes to the appropriate handler:

```rust
Commands::Think {
    query,
    protocol,
    profile,
    provider,
    model,
    ...
} => {
    // Create executor (mock or real)
    let executor = if mock {
        ProtocolExecutor::mock()?
    } else {
        let config = ExecutorConfig::default();
        config.llm.provider = provider.into();
        // ... configure
        ProtocolExecutor::with_config(config)?
    };

    // Execute protocol or profile
    let output = if let Some(proto) = protocol {
        executor.execute(&proto, input).await?
    } else {
        let prof = profile.unwrap_or("balanced".to_string());
        executor.execute_profile(&prof, input).await?
    };
}
```

**Key Files**:

- `src/main.rs` - CLI entry point
- `src/bin/mcp_cli.rs` - MCP-specific CLI commands

### 2.2 Input Construction: ProtocolInput

**Location**: `src/thinktool/executor.rs:306-365`

```rust
pub struct ProtocolInput {
    /// Input fields as key-value pairs
    pub fields: HashMap<String, serde_json::Value>,
}

impl ProtocolInput {
    pub fn query(query: impl Into<String>) -> Self { ... }
    pub fn argument(argument: impl Into<String>) -> Self { ... }
    pub fn statement(statement: impl Into<String>) -> Self { ... }
    pub fn claim(claim: impl Into<String>) -> Self { ... }
    pub fn work(work: impl Into<String>) -> Self { ... }
}
```

Input types map to ThinkTools:

- `query` -> GigaThink (expansive thinking)
- `argument` -> LaserLogic (deductive reasoning)
- `statement` -> BedRock (first principles)
- `claim` -> ProofGuard (verification)
- `work` -> BrutalHonesty (self-critique)

### 2.3 Protocol Executor: Core Orchestration

**Location**: `src/thinktool/executor.rs:423-469`

```rust
pub struct ProtocolExecutor {
    registry: ProtocolRegistry,      // Available protocols
    profiles: ProfileRegistry,       // Reasoning profiles
    config: ExecutorConfig,          // Execution settings
    llm_client: Option<UnifiedLlmClient>,  // LLM connection
}
```

#### 2.3.1 Protocol Loading

**Location**: `src/thinktool/registry.rs`

Protocols are loaded from:

1. `protocols/thinktools.yaml` (primary)
2. Hardcoded fallbacks (if YAML fails)
3. Custom TOML/JSON files

```rust
pub fn register_builtins(&mut self) -> Result<()> {
    // Try YAML first
    let yaml_path = cwd.join("protocols").join("thinktools.yaml");
    if yaml_path.exists() {
        self.load_from_yaml(&yaml_path)?;
    } else {
        // Fallback to hardcoded
        self.register(builtin_gigathink())?;
        self.register(builtin_laserlogic())?;
        // ...
    }
}
```

### 2.4 ThinkTool Selection

**Location**: `src/thinktool/executor.rs:487-595`

```rust
pub async fn execute(&self, protocol_id: &str, input: ProtocolInput) -> Result<ProtocolOutput> {
    // 1. Look up protocol in registry
    let protocol = self.registry
        .get(protocol_id)
        .ok_or_else(|| Error::NotFound { ... })?
        .clone();

    // 2. Validate input against protocol requirements
    self.validate_input(&protocol, &input)?;

    // 3. Initialize execution trace
    let mut trace = ExecutionTrace::new(&protocol.id, &protocol.version)
        .with_input(serde_json::to_value(&input.fields)?);

    // 4. Execute steps (sequential or parallel)
    let (step_results, step_outputs, total_tokens, step_traces) =
        if self.config.enable_parallel {
            self.execute_steps_parallel(&protocol.steps, &input, &start).await?
        } else {
            self.execute_steps_sequential(&protocol.steps, &input, &start).await?
        };

    // 5. Build output
    Ok(ProtocolOutput { ... })
}
```

### 2.5 Profile-Based Execution

**Location**: `src/engine/reasoning_loop.rs:82-153`

```rust
pub enum Profile {
    Quick,      // GigaThink -> LaserLogic (70% confidence)
    Balanced,   // GigaThink -> LaserLogic -> BedRock -> ProofGuard (80%)
    Deep,       // All 5 ThinkTools (85%)
    Paranoid,   // All 5 + second verification pass (95%)
}

impl Profile {
    pub fn thinktool_chain(&self) -> Vec<&'static str> {
        match self {
            Profile::Quick => vec!["gigathink", "laserlogic"],
            Profile::Balanced => vec!["gigathink", "laserlogic", "bedrock", "proofguard"],
            Profile::Deep => vec![
                "gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty"
            ],
            Profile::Paranoid => vec![
                "gigathink", "laserlogic", "bedrock", "proofguard",
                "brutalhonesty", "proofguard"  // Second verification
            ],
        }
    }
}
```

### 2.6 Step Execution

**Location**: `src/thinktool/executor.rs:598-684`

Sequential execution flow:

```rust
async fn execute_steps_sequential(&self, steps: &[ProtocolStep], ...) -> Result<...> {
    for (index, step) in steps.iter().enumerate() {
        // 1. Check dependencies
        if !self.dependencies_met(&step.depends_on, &step_results) {
            continue;
        }

        // 2. Check branch condition
        if let Some(condition) = &step.branch {
            if !self.evaluate_branch_condition(condition, &step_results) {
                traces.push(StepTrace { status: StepStatus::Skipped, ... });
                continue;
            }
        }

        // 3. Execute step
        let step_result = self.execute_step(step, input, &step_outputs, index).await?;

        // 4. Record trace
        let mut step_trace = StepTrace::new(&step.id, index);
        step_trace.complete(step_result.output.clone(), step_result.confidence);
        traces.push(step_trace);

        // 5. Store output for dependent steps
        step_outputs.insert(step.id.clone(), step_result.output.clone());
        step_results.push(step_result);
    }
}
```

### 2.7 LLM Integration

**Location**: `src/thinktool/llm.rs`

```rust
pub struct UnifiedLlmClient {
    config: LlmConfig,
    client: reqwest::Client,
}

// Supported providers (18+)
pub enum LlmProvider {
    Anthropic,     // Claude models
    OpenAI,        // GPT models
    GoogleGemini,  // Gemini models
    DeepSeek,      // DeepSeek V3/R1
    Groq,          // Fast inference
    // ... more
}

impl LlmClient for UnifiedLlmClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        match self.config.provider {
            LlmProvider::Anthropic => self.complete_anthropic(request).await,
            LlmProvider::OpenAI => self.complete_openai(request).await,
            // ...
        }
    }
}
```

### 2.8 ThinkTool Modules

**Location**: `src/thinktool/modules/`

| Module                | File                        | Purpose                                |
| --------------------- | --------------------------- | -------------------------------------- |
| GigaThink             | `gigathink.rs`              | 10+ perspective expansion              |
| LaserLogic            | `laserlogic.rs`             | Deductive reasoning, fallacy detection |
| BedRock               | `bedrock.rs`                | First principles decomposition         |
| ProofGuard            | `proofguard.rs`             | Multi-source verification              |
| BrutalHonesty         | `brutalhonesty.rs`          | Adversarial self-critique              |
| BrutalHonestyEnhanced | `brutalhonesty_enhanced.rs` | Cognitive bias detection               |

Each module implements:

```rust
pub trait ThinkToolModule: Send + Sync {
    fn config(&self) -> &ThinkToolModuleConfig;
    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput>;
}
```

### 2.9 Trace Capture

**Location**: `src/thinktool/trace.rs`

```rust
pub struct ExecutionTrace {
    pub id: Uuid,
    pub protocol_id: String,
    pub protocol_version: String,
    pub input: serde_json::Value,
    pub steps: Vec<StepTrace>,
    pub output: Option<serde_json::Value>,
    pub status: ExecutionStatus,
    pub timing: TimingInfo,
    pub tokens: TokenUsage,
    pub confidence: f64,
    pub metadata: TraceMetadata,
}

pub struct StepTrace {
    pub step_id: String,
    pub index: usize,
    pub prompt: String,
    pub raw_response: String,
    pub parsed_output: StepOutput,
    pub confidence: f64,
    pub duration_ms: u64,
    pub tokens: TokenUsage,
    pub status: StepStatus,
    pub error: Option<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}
```

### 2.10 M2 Connector (Experimental)

**Location**: `src/m2/`

The M2 connector provides integration with MiniMax M2 for interleaved thinking:

```rust
pub struct M2Connector {
    client: Client,
    config: M2Config,
}

impl M2Connector {
    pub async fn execute_interleaved_thinking(
        &self,
        protocol: &InterleavedProtocol,
        constraints: &CompositeConstraints,
        input: &ProtocolInput,
    ) -> Result<M2Result, Error> {
        // Currently stubbed - returns error for non-Ollama endpoints
    }
}
```

**Status**: M2 integration is experimental and not fully wired in.

### 2.11 Response Assembly

**Location**: `src/thinktool/executor.rs:583-595`

```rust
Ok(ProtocolOutput {
    protocol_id: protocol_id.to_string(),
    success,
    data,                  // Step outputs aggregated
    confidence,            // Average of step confidences
    steps: step_results,   // All step results
    tokens: total_tokens,  // Cumulative token usage
    duration_ms,           // Total execution time
    error,                 // Error message if failed
    trace_id,              // UUID for trace lookup
    budget_summary,        // Budget usage stats
})
```

---

## 3. Data Flow Diagram

```
Query Input
    |
    v
+---+-------------------+
|   ProtocolInput       |
|   fields: HashMap     |
+----------+------------+
           |
           v
+----------+------------+
|  ProtocolRegistry     |
|  get(protocol_id)     |
+----------+------------+
           |
           v
+----------+------------+
|    Protocol           |
|    - steps[]          |
|    - strategy         |
|    - input/output     |
+----------+------------+
           |
           v
+----------+------------+
|  For each step:       |
|  1. Check deps        |
|  2. Check branch      |
|  3. Render template   |
|  4. Build system msg  |
|  5. Call LLM          |
|  6. Parse output      |
|  7. Record trace      |
+----------+------------+
           |
           v
+----------+------------+
|  ExecutionTrace       |
|  - steps[]            |
|  - timing             |
|  - tokens             |
+----------+------------+
           |
           v
+----------+------------+
|  ProtocolOutput       |
|  - confidence         |
|  - steps              |
|  - data               |
+----------+------------+
```

---

## 4. Gaps and Issues Identified

### 4.1 M2 Integration Not Implemented

**Location**: `src/m2/mod.rs:47-57`

```rust
pub async fn execute_for_use_case(...) -> Result<InterleavedResult, Error> {
    Err(Error::M2ExecutionError(
        "M2 integration is not implemented in this build".to_string(),
    ))
}
```

**Impact**: M2 interleaved thinking features are unavailable.
**Recommendation**: Complete M2 integration or remove from public API.

### 4.2 Memory Feature Gate Required

Memory integration requires the `memory` feature flag:

```rust
#[cfg(feature = "memory")]
pub mod rag;
```

**Impact**: RAG capabilities not available in default build.
**Recommendation**: Document clearly in user guides.

### 4.3 Trace Persistence Optional

Traces are only saved when `save_traces` is true:

```rust
let trace_id = if self.config.save_traces {
    self.save_trace(&trace)?;
    Some(trace.id.to_string())
} else {
    None
};
```

**Impact**: Auditability requires explicit configuration.
**Recommendation**: Consider enabling by default for production.

### 4.4 Parallel Execution Error Handling

**Location**: `src/thinktool/executor.rs:689-923`

Parallel execution creates new LLM clients per step:

```rust
let llm_client = self.llm_client.as_ref().map(|_| {
    UnifiedLlmClient::new(config.llm.clone()).ok()
});
```

**Impact**: May create connection overhead.
**Recommendation**: Consider connection pooling.

### 4.5 Mock Mode Confidence Values

Mock mode returns fixed confidence values:

```rust
StepAction::Generate { min_count, .. } => {
    format!("{}\n\nConfidence: 0.85", items.join("\n"))
}
```

**Impact**: Testing may not reflect real-world variance.
**Recommendation**: Consider randomized mock responses.

---

## 5. Test Recommendations

### 5.1 Full Workflow Integration Test

```rust
#[tokio::test]
async fn test_full_reason_workflow() {
    // 1. Create executor with real or mock LLM
    let executor = ProtocolExecutor::with_config(ExecutorConfig {
        use_mock: true,
        save_traces: true,
        trace_dir: Some(PathBuf::from("/tmp/test_traces")),
        ..Default::default()
    })?;

    // 2. Create input
    let input = ProtocolInput::query("What factors contribute to startup success?");

    // 3. Execute with balanced profile
    let output = executor.execute_profile("balanced", input).await?;

    // 4. Verify output structure
    assert!(output.success);
    assert!(output.confidence >= 0.80); // Balanced threshold
    assert!(!output.steps.is_empty());
    assert!(output.trace_id.is_some());

    // 5. Verify step sequence
    let step_ids: Vec<&str> = output.steps.iter()
        .map(|s| s.step_id.as_str())
        .collect();

    // Should include steps from gigathink, laserlogic, bedrock, proofguard
    assert!(step_ids.iter().any(|id| id.contains("identify") || id.contains("generate")));
    assert!(step_ids.iter().any(|id| id.contains("extract") || id.contains("analyze")));

    // 6. Verify trace was saved
    let trace_path = PathBuf::from("/tmp/test_traces");
    let trace_files: Vec<_> = std::fs::read_dir(&trace_path)?
        .filter_map(|e| e.ok())
        .collect();
    assert!(!trace_files.is_empty());
}
```

### 5.2 Profile Escalation Test

```rust
#[tokio::test]
async fn test_profile_escalation() {
    let executor = ProtocolExecutor::mock()?;
    let input = ProtocolInput::query("Complex decision requiring deep analysis");

    // Test each profile
    for profile in &["quick", "balanced", "deep", "paranoid"] {
        let output = executor.execute_profile(profile, input.clone()).await?;

        let expected_threshold = match *profile {
            "quick" => 0.70,
            "balanced" => 0.80,
            "deep" => 0.85,
            "paranoid" => 0.95,
            _ => 0.0,
        };

        // Mock always succeeds, but step count should increase
        assert!(output.success);
        println!("{}: {} steps, {:.2} confidence",
            profile, output.steps.len(), output.confidence);
    }
}
```

### 5.3 Trace Verification Test

```rust
#[tokio::test]
async fn test_trace_capture_complete() {
    let executor = ProtocolExecutor::with_config(ExecutorConfig {
        use_mock: true,
        save_traces: true,
        trace_dir: Some(PathBuf::from("/tmp/trace_test")),
        ..Default::default()
    })?;

    let input = ProtocolInput::query("Test query for trace verification");
    let output = executor.execute("gigathink", input).await?;

    // Read saved trace
    let trace_id = output.trace_id.expect("Trace should be saved");
    let trace_path = format!("/tmp/trace_test/gigathink_{}.json", trace_id);
    let trace_content = std::fs::read_to_string(&trace_path)?;
    let trace: ExecutionTrace = serde_json::from_str(&trace_content)?;

    // Verify trace completeness
    assert_eq!(trace.protocol_id, "gigathink");
    assert_eq!(trace.status, ExecutionStatus::Completed);
    assert!(!trace.steps.is_empty());

    for step in &trace.steps {
        assert!(!step.step_id.is_empty());
        assert!(step.duration_ms > 0);
        assert!(step.confidence > 0.0);
    }
}
```

---

## 6. Key File Reference

| Purpose           | File Path                                |
| ----------------- | ---------------------------------------- |
| CLI Entry         | `src/main.rs`                            |
| Executor Core     | `src/thinktool/executor.rs`              |
| Protocol Registry | `src/thinktool/registry.rs`              |
| Profile System    | `src/thinktool/profiles.rs`              |
| LLM Client        | `src/thinktool/llm.rs`                   |
| Trace Types       | `src/thinktool/trace.rs`                 |
| Module Trait      | `src/thinktool/modules/mod.rs`           |
| GigaThink         | `src/thinktool/modules/gigathink.rs`     |
| LaserLogic        | `src/thinktool/modules/laserlogic.rs`    |
| BedRock           | `src/thinktool/modules/bedrock.rs`       |
| ProofGuard        | `src/thinktool/modules/proofguard.rs`    |
| BrutalHonesty     | `src/thinktool/modules/brutalhonesty.rs` |
| M2 Connector      | `src/m2/connector.rs`                    |
| ReasoningLoop     | `src/engine/reasoning_loop.rs`           |
| Protocol Loader   | `src/thinktool/yaml_loader.rs`           |
| Integration Tests | `tests/thinktool_integration_tests.rs`   |

---

## 7. Conclusion

The Query->Reason->Response workflow in ReasonKit Core is well-architected with clear separation of concerns:

1. **Entry**: CLI/API receives query and routes to executor
2. **Selection**: Protocol/Profile registry determines ThinkTool chain
3. **Execution**: Steps execute sequentially/parallel with LLM calls
4. **Tracing**: Every step is captured for auditability
5. **Response**: Aggregated results with confidence scores

**Strengths**:

- Clean module separation
- Comprehensive trace capture
- Flexible profile system
- Multiple LLM provider support (18+)
- Parallel execution capability

**Areas for Improvement**:

- Complete M2 integration
- Connection pooling for parallel execution
- More realistic mock mode
- Default trace persistence

---

_Document generated: 2026-01-01_
_ReasonKit Core Version: 0.1.x_
