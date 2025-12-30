# ThinkTool Protocol Engine Design

> "Turn Prompts into Protocols" - The Core Value Proposition

**Version**: 1.0.0
**Status**: Design Document
**Author**: ReasonKit Team
**Date**: 2025-12-11

---

## Executive Summary

ThinkTools are **structured reasoning protocols** that transform ad-hoc LLM prompting into auditable, reproducible reasoning chains. Unlike RAG (which retrieves information) or multi-agent orchestration (which coordinates), ThinkTools define **HOW** an LLM should reason about a problem.

This document specifies the Protocol Engine that executes ThinkTool protocols.

---

## 1. Core Concept: What is a ThinkTool?

### 1.1 Definition

A **ThinkTool** is a structured reasoning protocol specification that:

1. **Defines a reasoning strategy** (e.g., expansive, deductive, adversarial)
2. **Structures the thought process** into auditable steps
3. **Produces verifiable output** with confidence scores
4. **Maintains provenance** of reasoning chain

### 1.2 ThinkTool vs Traditional Prompting

| Aspect          | Traditional Prompting | ThinkTool Protocol         |
| --------------- | --------------------- | -------------------------- |
| Structure       | Free-form text        | Typed schema               |
| Auditability    | Opaque                | Step-by-step trace         |
| Reproducibility | Low                   | Deterministic given inputs |
| Composition     | Manual                | Protocol chaining          |
| Verification    | None                  | Built-in confidence        |

### 1.3 Open Source ThinkTools (reasonkit-core)

| Tool              | Code | Purpose                        | Strategy                      |
| ----------------- | ---- | ------------------------------ | ----------------------------- |
| **GigaThink**     | `gt` | Expansive creative thinking    | Generate 10+ perspectives     |
| **LaserLogic**    | `ll` | Precision deductive reasoning  | Fallacy detection, syllogisms |
| **BedRock**       | `br` | First principles decomposition | Axiom identification          |
| **ProofGuard**    | `pg` | Multi-source verification      | Triangulation protocol        |
| **BrutalHonesty** | `bh` | Adversarial self-critique      | Find flaws first              |

---

## 2. Protocol Specification Language

### 2.1 Protocol Schema

```rust
/// A ThinkTool Protocol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    /// Unique protocol identifier (e.g., "gigathink", "laserlogic")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Protocol version (semver)
    pub version: String,

    /// Brief description
    pub description: String,

    /// Reasoning strategy category
    pub strategy: ReasoningStrategy,

    /// Input specification
    pub input: InputSpec,

    /// Protocol steps (ordered)
    pub steps: Vec<ProtocolStep>,

    /// Output specification
    pub output: OutputSpec,

    /// Validation rules
    pub validation: Vec<ValidationRule>,

    /// Metadata for composition
    pub metadata: ProtocolMetadata,
}

/// Reasoning strategy categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    /// Divergent thinking - maximize perspectives
    Expansive,
    /// Convergent thinking - deduce conclusions
    Deductive,
    /// Break down to fundamentals
    Analytical,
    /// Challenge and critique
    Adversarial,
    /// Cross-reference and confirm
    Verification,
    /// Weigh options systematically
    Decision,
    /// Scientific method
    Empirical,
}
```

### 2.2 Protocol Step Definition

```rust
/// A single step in a protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStep {
    /// Step identifier within protocol
    pub id: String,

    /// What this step does
    pub action: StepAction,

    /// Prompt template (with placeholders)
    pub prompt_template: String,

    /// Expected output format
    pub output_format: OutputFormat,

    /// Minimum confidence to proceed
    pub min_confidence: f64,

    /// Dependencies on previous steps
    pub depends_on: Vec<String>,

    /// Optional branching conditions
    pub branch: Option<BranchCondition>,
}

/// Step action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepAction {
    /// Generate perspectives/ideas
    Generate { min_count: usize, max_count: usize },
    /// Analyze/evaluate input
    Analyze { criteria: Vec<String> },
    /// Synthesize multiple inputs
    Synthesize { aggregation: AggregationType },
    /// Validate against rules
    Validate { rules: Vec<String> },
    /// Challenge/critique
    Critique { severity: CritiqueSeverity },
    /// Make decision
    Decide { method: DecisionMethod },
    /// Cross-reference sources
    CrossReference { min_sources: usize },
}
```

### 2.3 Example: GigaThink Protocol Definition

```json
{
  "id": "gigathink",
  "name": "GigaThink",
  "version": "1.0.0",
  "description": "Expansive creative thinking - generate 10+ diverse perspectives",
  "strategy": "Expansive",
  "input": {
    "required": ["query"],
    "optional": ["context", "constraints"]
  },
  "steps": [
    {
      "id": "identify_dimensions",
      "action": { "Generate": { "min_count": 5, "max_count": 10 } },
      "prompt_template": "Identify 5-10 distinct dimensions/angles to analyze: {{query}}\nContext: {{context}}",
      "output_format": "List",
      "min_confidence": 0.7
    },
    {
      "id": "explore_each",
      "action": {
        "Analyze": { "criteria": ["novelty", "relevance", "depth"] }
      },
      "prompt_template": "For dimension '{{dimension}}', provide:\n1. Key insight\n2. Evidence/example\n3. Implications\n4. Confidence (0-1)",
      "output_format": "Structured",
      "min_confidence": 0.6,
      "depends_on": ["identify_dimensions"]
    },
    {
      "id": "synthesize",
      "action": { "Synthesize": { "aggregation": "ThematicClustering" } },
      "prompt_template": "Synthesize the {{count}} perspectives into key themes and actionable insights",
      "output_format": "Structured",
      "min_confidence": 0.8,
      "depends_on": ["explore_each"]
    },
    {
      "id": "confidence_calibration",
      "action": {
        "Validate": { "rules": ["coherence", "coverage", "novelty"] }
      },
      "prompt_template": "Rate overall analysis confidence considering:\n- Coverage of solution space\n- Coherence of perspectives\n- Novelty of insights",
      "output_format": "Score",
      "min_confidence": 0.7,
      "depends_on": ["synthesize"]
    }
  ],
  "output": {
    "format": "GigaThinkResult",
    "fields": ["perspectives", "themes", "synthesis", "confidence"]
  },
  "validation": [
    { "rule": "min_perspectives", "value": 5 },
    { "rule": "confidence_range", "min": 0.0, "max": 1.0 }
  ],
  "metadata": {
    "category": "creative",
    "composable_with": ["laserlogic", "brutalhonesty"],
    "typical_tokens": 2000,
    "estimated_latency_ms": 5000
  }
}
```

---

## 3. Protocol Engine Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THINKTOOL PROTOCOL ENGINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Protocol   │    │   Executor   │    │   Output     │          │
│  │   Registry   │───▶│    Core      │───▶│  Validator   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Protocol   │    │    Step      │    │   Trace      │          │
│  │   Loader     │    │  Scheduler   │    │   Store      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                                       │
│                             ▼                                       │
│                      ┌──────────────┐                               │
│                      │ LLM Provider │                               │
│                      │   Adapter    │                               │
│                      └──────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### Protocol Registry

```rust
/// Registry of available ThinkTool protocols
pub struct ProtocolRegistry {
    /// Loaded protocols by ID
    protocols: HashMap<String, Protocol>,
    /// Protocol search paths
    search_paths: Vec<PathBuf>,
}

impl ProtocolRegistry {
    /// Load all protocols from configured paths
    pub fn load_all(&mut self) -> Result<usize>;

    /// Get protocol by ID
    pub fn get(&self, id: &str) -> Option<&Protocol>;

    /// List all available protocols
    pub fn list(&self) -> Vec<&Protocol>;

    /// Register a new protocol
    pub fn register(&mut self, protocol: Protocol) -> Result<()>;
}
```

#### Executor Core

```rust
/// Executes ThinkTool protocols
pub struct ProtocolExecutor {
    /// Protocol registry
    registry: ProtocolRegistry,
    /// LLM provider
    llm: Box<dyn LlmProvider>,
    /// Configuration
    config: ExecutorConfig,
    /// Trace storage
    trace_store: TraceStore,
}

impl ProtocolExecutor {
    /// Execute a protocol with given input
    pub async fn execute(
        &self,
        protocol_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput>;

    /// Execute a protocol chain (composition)
    pub async fn execute_chain(
        &self,
        chain: &[ChainStep],
        input: ProtocolInput,
    ) -> Result<ProtocolOutput>;

    /// Resume from checkpoint
    pub async fn resume(
        &self,
        trace_id: &str,
        from_step: &str,
    ) -> Result<ProtocolOutput>;
}
```

#### Trace Store (Auditability)

```rust
/// Stores execution traces for auditability
pub struct TraceStore {
    /// Storage backend
    storage: Box<dyn TraceStorage>,
}

/// A single execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Unique trace ID
    pub id: Uuid,
    /// Protocol that was executed
    pub protocol_id: String,
    /// Input provided
    pub input: ProtocolInput,
    /// Step-by-step execution record
    pub steps: Vec<StepTrace>,
    /// Final output
    pub output: Option<ProtocolOutput>,
    /// Overall status
    pub status: ExecutionStatus,
    /// Timing information
    pub timing: TimingInfo,
    /// Token usage
    pub tokens: TokenUsage,
}

/// Trace of a single step execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTrace {
    /// Step ID
    pub step_id: String,
    /// Actual prompt sent to LLM
    pub prompt: String,
    /// Raw LLM response
    pub raw_response: String,
    /// Parsed/structured response
    pub parsed_output: serde_json::Value,
    /// Confidence score
    pub confidence: f64,
    /// Step timing
    pub duration_ms: u64,
    /// Tokens used
    pub tokens: TokenUsage,
}
```

---

## 4. Protocol Composition (Profiles)

### 4.1 Reasoning Profiles

Profiles compose multiple ThinkTools into coherent reasoning chains:

```rust
/// A reasoning profile (composition of protocols)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningProfile {
    /// Profile identifier (e.g., "quick", "balanced", "paranoid")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Protocols in execution order
    pub chain: Vec<ChainStep>,
    /// Expected confidence threshold
    pub min_confidence: f64,
    /// Typical token budget
    pub token_budget: Option<u32>,
}

/// A step in a protocol chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    /// Protocol to execute
    pub protocol_id: String,
    /// Input mapping from previous step
    pub input_mapping: HashMap<String, String>,
    /// Condition to execute (optional)
    pub condition: Option<ChainCondition>,
    /// Override configuration
    pub config_override: Option<serde_json::Value>,
}
```

### 4.2 Built-in Profiles

| Profile        | Chain                       | Min Confidence | Use Case                |
| -------------- | --------------------------- | -------------- | ----------------------- |
| `--quick`      | gt → ll                     | 0.70           | Fast 2-step analysis    |
| `--balanced`   | gt → ll → br → pg           | 0.80           | Standard 4-module chain |
| `--deep`       | gt → ll → br → pg → hr      | 0.85           | Thorough analysis       |
| `--paranoid`   | gt → ll → br → pg → bh → rr | 0.95           | Maximum verification    |
| `--decide`     | dm → rr → ll → hr           | 0.85           | Decision support        |
| `--scientific` | se → ab → br → pg           | 0.85           | Research & experiments  |

### 4.3 Profile Definition Example

```json
{
  "id": "balanced",
  "name": "Balanced Analysis",
  "chain": [
    {
      "protocol_id": "gigathink",
      "input_mapping": { "query": "input.query" }
    },
    {
      "protocol_id": "laserlogic",
      "input_mapping": {
        "claims": "steps.gigathink.perspectives",
        "context": "input.context"
      }
    },
    {
      "protocol_id": "bedrock",
      "input_mapping": {
        "statement": "steps.laserlogic.conclusion",
        "assumptions": "steps.laserlogic.assumptions"
      },
      "condition": { "confidence_below": 0.9 }
    },
    {
      "protocol_id": "proofguard",
      "input_mapping": {
        "claims": "steps.bedrock.axioms",
        "sources": "input.sources"
      }
    }
  ],
  "min_confidence": 0.8,
  "token_budget": 8000
}
```

---

## 5. CLI Interface

### 5.1 Command Structure

```bash
# Execute a single protocol
rk think <QUERY> --protocol <PROTOCOL_ID> [OPTIONS]

# Execute a reasoning profile
rk think <QUERY> --profile <balanced|quick|deep|paranoid|decide|scientific>

# Short aliases
rk gt "Analyze the market opportunity"        # GigaThink
rk ll "Evaluate this argument: X implies Y"  # LaserLogic
rk br "What are the first principles of..."   # BedRock
rk pg "Verify: claim X with sources Y, Z"    # ProofGuard
rk bh "Critique this plan: ..."              # BrutalHonesty

# Profile shortcuts
rk think "Complex question" --quick      # Fast analysis
rk think "Important decision" --paranoid # Maximum rigor
```

### 5.2 Output Modes

```bash
# Default: Summary output
rk think "Query" --profile balanced

# Verbose: Full reasoning trace
rk think "Query" --profile balanced --verbose

# JSON: Machine-readable
rk think "Query" --profile balanced --json

# Trace: Save execution for audit
rk think "Query" --profile balanced --save-trace ./traces/
```

### 5.3 Example CLI Session

```
$ rk think "Should we adopt microservices architecture?" --profile balanced --verbose

═══════════════════════════════════════════════════════════════════════
  REASONKIT PROTOCOL ENGINE v1.0
  Profile: balanced (GigaThink → LaserLogic → BedRock → ProofGuard)
═══════════════════════════════════════════════════════════════════════

[1/4] GigaThink: Generating perspectives...
  ├─ Dimension 1: Scalability (conf: 0.92)
  ├─ Dimension 2: Team structure (conf: 0.88)
  ├─ Dimension 3: Operational complexity (conf: 0.85)
  ├─ Dimension 4: Development velocity (conf: 0.79)
  ├─ Dimension 5: Cost implications (conf: 0.91)
  ├─ Dimension 6: Fault isolation (conf: 0.87)
  ├─ Dimension 7: Technology flexibility (conf: 0.83)
  └─ Step confidence: 0.86

[2/4] LaserLogic: Analyzing logical structure...
  ├─ Premise 1: "Large team" → supports microservices
  ├─ Premise 2: "Need independent scaling" → supports microservices
  ├─ Counter: "Distributed systems complexity" → concerns
  ├─ Fallacy check: None detected
  └─ Step confidence: 0.82

[3/4] BedRock: First principles decomposition...
  ├─ Axiom 1: System boundaries should match team boundaries
  ├─ Axiom 2: Coupling and cohesion trade-offs
  ├─ Axiom 3: Operational capability requirements
  └─ Step confidence: 0.88

[4/4] ProofGuard: Cross-reference verification...
  ├─ Source 1: [✓] Martin Fowler's microservices prerequisites
  ├─ Source 2: [✓] Netflix engineering blog
  ├─ Source 3: [✓] Domain-Driven Design patterns
  └─ Step confidence: 0.90

═══════════════════════════════════════════════════════════════════════
  SYNTHESIS
═══════════════════════════════════════════════════════════════════════

RECOMMENDATION: Conditional adoption with prerequisites

Key factors:
  1. Team size and structure readiness (critical)
  2. Operational maturity for distributed systems (critical)
  3. Clear domain boundaries identified (important)

Confidence: 0.85 (balanced profile threshold: 0.80) ✓

Execution: 4 steps | 3,247 tokens | 4.2s
Trace saved: ./traces/balanced_20251211_143052.json
```

---

## 6. Implementation Roadmap

### Phase 1: Core Engine (MVP)

1. Protocol schema definition (Rust types)
2. Protocol loader (JSON/YAML files)
3. Single protocol executor
4. Basic CLI (`rk think --protocol`)
5. GigaThink protocol implementation

### Phase 2: Composition

1. Profile system
2. Chain executor
3. Built-in profiles (quick, balanced, deep, paranoid)
4. All 5 OSS protocols

### Phase 3: Auditability

1. Trace store
2. Trace viewer CLI
3. Export formats (JSON, Markdown)
4. Resume from checkpoint

### Phase 4: Integration

1. RAG integration (retrieve → think)
2. Multi-provider LLM support
3. Cost optimization
4. Streaming output

---

## 7. Rust Module Structure

```
src/
├── thinktool/
│   ├── mod.rs              # Module exports
│   ├── protocol.rs         # Protocol types and schema
│   ├── registry.rs         # Protocol registry
│   ├── executor.rs         # Protocol executor
│   ├── step.rs             # Step execution
│   ├── trace.rs            # Execution trace types
│   ├── output.rs           # Output formatting
│   └── profiles/
│       ├── mod.rs
│       ├── quick.rs
│       ├── balanced.rs
│       ├── deep.rs
│       └── paranoid.rs
├── protocols/              # Built-in protocol definitions
│   ├── gigathink.json
│   ├── laserlogic.json
│   ├── bedrock.json
│   ├── proofguard.json
│   └── brutalhonesty.json
└── cli/
    └── think.rs            # `rk think` command
```

---

## 8. Key Design Decisions

### 8.1 Why JSON/YAML for Protocol Definitions?

1. **Human-readable**: Non-programmers can understand/modify
2. **Versionable**: Easy to diff, review, track changes
3. **Portable**: Can be shared, published, composed
4. **Runtime loading**: No recompilation for new protocols

### 8.2 Why Rust for the Engine?

1. **Performance**: Sub-5ms overhead per step
2. **Type safety**: Protocol schema validation at compile time
3. **Async**: Parallel step execution where possible
4. **Memory safety**: No leaks in long-running processes

### 8.3 Why Trace Everything?

1. **Auditability**: Know exactly how conclusions were reached
2. **Debugging**: Identify where reasoning went wrong
3. **Reproducibility**: Re-run with same inputs
4. **Learning**: Improve protocols based on traces

---

## 9. Success Metrics

| Metric                      | Target                  | Measurement       |
| --------------------------- | ----------------------- | ----------------- |
| Protocol execution overhead | < 50ms                  | Benchmark         |
| Step-to-step latency        | < 10ms                  | Profiling         |
| Trace storage efficiency    | < 1KB/step              | Size audit        |
| Protocol load time          | < 100ms                 | Startup benchmark |
| User satisfaction           | > 80% prefer structured | Survey            |

---

## 10. References

- ORCHESTRATOR.md - ThinkTool definitions
- chain_of_thought_v3_lite.json - Token-optimized schema
- Reflexion paper (Shinn et al., 2023) - Self-reflection patterns
- Tree of Thoughts (Yao et al., 2023) - Deliberate reasoning
- Constitutional AI (Anthropic, 2022) - Structured critique

---

_ThinkTool Protocol Engine v1.0 | "Turn Prompts into Protocols"_
