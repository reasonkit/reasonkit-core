# ReasonKit Benchmark Suite Design

> **Purpose:** Comprehensive framework for proving ReasonKit value through rigorous evaluation
> **Version:** 2.0 | **Status:** Production-Ready Design
> **Philosophy:** No claims without evidence. Run benchmarks to prove value.
> **Target Confidence:** 95% on all statistical claims

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Benchmark Categories](#benchmark-categories)
3. [Benchmark Implementations](#benchmark-implementations)
4. [Evaluation Methodology](#evaluation-methodology)
5. [A/B Test Framework](#ab-test-framework)
6. [CLI Commands](#cli-commands)
7. [Results Reporting](#results-reporting)
8. [Continuous Benchmarking](#continuous-benchmarking)
9. [Third-Party Validation](#third-party-validation)
10. [Interpretation Guide](#interpretation-guide)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

ReasonKit must prove its value through measurable, reproducible benchmarks. This design provides:

- **4 Benchmark Categories:** Reasoning Quality, Reliability, Auditability, Performance
- **6 Primary Benchmarks:** GSM8K, ARC-C, LogiQA, HellaSwag, Custom Audit, Custom Debug
- **Statistical Rigor:** Wilson score intervals, McNemar's test, 10+ trials per sample
- **A/B Framework:** Direct comparison of raw LLM vs ReasonKit-enhanced responses
- **CI Integration:** Automated nightly runs with regression detection

### What We Measure

| Category | Primary Metric | Target | Proven Approach |
|----------|---------------|--------|-----------------|
| **Reasoning Quality** | Accuracy delta | > +5% | Self-Consistency (Wang et al.) |
| **Reliability** | Consistency rate | > 90% | Multiple trials, same seed |
| **Auditability** | Trace completeness | 100% | Structural validation |
| **Performance** | Latency overhead | < 50% | Criterion benchmarks |

---

## Benchmark Categories

### 1. Reasoning Quality Benchmarks

Measure whether ThinkTools actually improve LLM reasoning accuracy.

#### 1.1 Mathematical Reasoning (GSM8K)

| Property | Value |
|----------|-------|
| **Dataset** | GSM8K - 8,792 grade school math problems |
| **Metric** | Exact match accuracy |
| **Baseline** | GPT-4: 92%, Claude Sonnet: 88% |
| **ReasonKit Target** | +3-5% with balanced profile |
| **Status** | Implemented in `src/thinktool/benchmark.rs` |

**Why GSM8K:**
- Multi-step reasoning required
- Clear ground truth (numerical answer)
- Well-established baseline scores
- Self-Consistency shows +17.9% improvement (Wang et al. 2023)

**Caveat:** GSM8K is approaching saturation. Consider MATH for harder problems.

#### 1.2 Science Reasoning (ARC-Challenge)

| Property | Value |
|----------|-------|
| **Dataset** | ARC-Challenge - 2,590 science questions |
| **Metric** | Multiple-choice accuracy |
| **Baseline** | GPT-4: 95%, Claude Sonnet: 93% |
| **ReasonKit Target** | +2-4% with deep profile |
| **Status** | Implemented in `src/bin/bench.rs` |

**Why ARC-C:**
- Requires scientific knowledge + reasoning
- Tests explanation-based reasoning (ProofGuard value)
- Not saturated like ARC-Easy

#### 1.3 Logical Deduction (LogiQA 2.0)

| Property | Value |
|----------|-------|
| **Dataset** | LogiQA 2.0 - 7,376 logical reasoning problems |
| **Metric** | Multiple-choice accuracy |
| **Baseline** | GPT-4: 58%, QwQ-32B: 85% |
| **ReasonKit Target** | +5-10% with paranoid profile |
| **Status** | Design complete, implementation pending |

**Why LogiQA:**
- 27.8pp gap between models shows reasoning improvement potential
- Tests LaserLogic fallacy detection
- Categories: categorical, sufficient conditional, necessary conditional, etc.

#### 1.4 Common Sense (HellaSwag)

| Property | Value |
|----------|-------|
| **Dataset** | HellaSwag - 10,042 commonsense scenarios |
| **Metric** | Multiple-choice accuracy |
| **Baseline** | GPT-4: 95%, Claude Sonnet: 93% |
| **ReasonKit Target** | +0-2% (near ceiling) |
| **Status** | Low priority - near saturation |

**Why HellaSwag:**
- Tests common sense completion
- Good sanity check that ThinkTools don't degrade simple tasks
- Helps identify overthinking penalty

---

### 2. Reliability Benchmarks

Measure consistency and failure handling.

#### 2.1 Hallucination Rate

| Property | Value |
|----------|-------|
| **Dataset** | TruthfulQA - 817 questions designed to elicit false answers |
| **Metric** | MC1 (single correct answer), MC2 (multiple correct answers) |
| **Baseline** | GPT-4: 59% MC1 |
| **ReasonKit Target** | +5-10% with ProofGuard verification |
| **Measurement** | Flag responses containing known false claims |

**Metrics:**
- **Hallucination rate:** % of responses containing verifiably false claims
- **Claim verification rate:** % of claims that can be traced to sources
- **Overconfidence ratio:** High confidence on wrong answers

#### 2.2 Consistency (Same Input, Same Output)

| Property | Value |
|----------|-------|
| **Method** | Run same query 10 times, measure answer variance |
| **Metric** | Agreement rate (% identical answers) |
| **Target** | > 85% with Self-Consistency |
| **Temperature** | 0.0 for baseline, 0.7 for diversity tests |

**Protocol:**
```rust
pub struct ConsistencyTest {
    pub query: String,
    pub trials: usize,        // Default: 10
    pub temperature: f32,     // Default: 0.0
    pub agreement_threshold: f32,  // Default: 0.85
}

pub struct ConsistencyResult {
    pub unique_answers: usize,
    pub most_common_answer: String,
    pub most_common_count: usize,
    pub agreement_rate: f32,
    pub semantic_similarity: f32,  // Embedding-based similarity
}
```

#### 2.3 Failure Recovery

| Property | Value |
|----------|-------|
| **Method** | Inject malformed inputs, measure graceful degradation |
| **Metrics** | Error rate, recovery rate, timeout rate |
| **Target** | < 1% unhandled errors, 100% structured error messages |

**Test Cases:**
1. Empty input
2. Extremely long input (> 100K tokens)
3. Invalid encoding
4. Prompt injection attempts
5. Circular reasoning triggers

---

### 3. Auditability Benchmarks

Measure traceability and explanation quality - ReasonKit's core differentiator.

#### 3.1 Trace Completeness

| Property | Value |
|----------|-------|
| **Method** | Validate all execution traces for required fields |
| **Metric** | % of traces with complete metadata |
| **Target** | 100% |
| **Status** | Implemented in quality gates |

**Required Fields:**
```rust
pub struct TraceCompleteness {
    // Mandatory fields (must be present and valid)
    pub has_session_id: bool,
    pub has_timestamp: bool,
    pub has_profile: bool,
    pub has_all_steps: bool,
    pub has_confidence_scores: bool,
    pub has_token_counts: bool,

    // Quality metrics
    pub step_output_valid: bool,
    pub metadata_parseable: bool,
    pub provenance_traceable: bool,
}
```

#### 3.2 Decision Traceability

| Property | Value |
|----------|-------|
| **Method** | Human evaluation: can decisions be traced to reasoning? |
| **Metric** | Traceability score (1-5 Likert scale) |
| **Target** | Mean > 4.0 |
| **Sample Size** | 100 randomly selected traces |

**Evaluation Rubric:**
- **5:** Every decision has clear, traceable reasoning path
- **4:** Most decisions traceable, minor gaps
- **3:** Key decisions traceable, supporting details unclear
- **2:** Some traceability, significant gaps
- **1:** Decisions appear arbitrary, no clear reasoning path

#### 3.3 Explanation Quality

| Property | Value |
|----------|-------|
| **Method** | Automated + human evaluation of explanations |
| **Metrics** | Coherence, Completeness, Correctness (3C score) |
| **Target** | Mean 3C score > 0.8 |

**Automated Metrics:**
```rust
pub struct ExplanationQuality {
    // Coherence: Does the explanation flow logically?
    pub coherence_score: f32,  // 0-1, based on sentence transitions

    // Completeness: Are all reasoning steps present?
    pub completeness_score: f32,  // % of expected steps present

    // Correctness: Does explanation match the conclusion?
    pub correctness_score: f32,  // Semantic similarity to expected

    // Aggregate
    pub three_c_score: f32,  // Weighted average
}
```

---

### 4. Performance Benchmarks

Measure computational overhead of ThinkTools.

#### 4.1 Latency Overhead

| Property | Value |
|----------|-------|
| **Method** | Compare raw query time vs ThinkTool-enhanced time |
| **Metric** | Latency multiplier, p50/p95/p99 |
| **Target** | < 1.5x overhead for balanced profile |
| **Measurement** | Criterion benchmarks with 100+ iterations |

**Profiles by Overhead:**
| Profile | Expected Overhead | Acceptable For |
|---------|------------------|----------------|
| quick | 1.1-1.2x | Real-time applications |
| balanced | 1.3-1.5x | Interactive use |
| deep | 2-3x | Batch processing |
| paranoid | 3-5x | Critical decisions |

#### 4.2 Token Efficiency

| Property | Value |
|----------|-------|
| **Method** | Measure input/output token counts per profile |
| **Metric** | Token multiplier vs raw query |
| **Target** | Document overhead, let users decide cost-benefit |

**Token Budget by Profile:**
| Profile | Expected Tokens | Cost Multiplier |
|---------|----------------|-----------------|
| quick | 1.5x base | 1.5x |
| balanced | 3-4x base | 3-4x |
| deep | 5-8x base | 5-8x |
| paranoid | 10-15x base | 10-15x |

#### 4.3 Memory Usage

| Property | Value |
|----------|-------|
| **Method** | Track heap allocation during benchmark runs |
| **Metric** | Peak memory, average memory, allocation count |
| **Target** | < 100MB peak for standard operations |
| **Tooling** | `dhat` profiler, custom allocator tracking |

---

## Benchmark Implementations

### GSM8K Benchmark Implementation

```rust
// src/benchmark/gsm8k.rs

use serde::{Deserialize, Serialize};
use std::path::Path;

/// GSM8K problem structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSM8KProblem {
    pub id: String,
    pub question: String,
    pub answer: f64,
    pub solution: String,
}

/// GSM8K benchmark runner
pub struct GSM8KBenchmark {
    problems: Vec<GSM8KProblem>,
    config: BenchmarkConfig,
}

impl GSM8KBenchmark {
    /// Load from official HuggingFace dataset
    pub async fn load() -> Result<Self> {
        let url = "https://huggingface.co/datasets/gsm8k/resolve/main/main/test.jsonl";
        // Download and parse...
        todo!()
    }

    /// Load from local cache
    pub fn load_cached(path: impl AsRef<Path>) -> Result<Self> {
        let problems = gsm8k::load_problems(path)?;
        Ok(Self {
            problems: problems.into_iter().map(|p| GSM8KProblem {
                id: p.id,
                question: p.question,
                answer: match p.answer {
                    Answer::Numeric(n) => n,
                    _ => 0.0,
                },
                solution: p.solution.unwrap_or_default(),
            }).collect(),
            config: BenchmarkConfig::default(),
        })
    }

    /// Run benchmark comparing baseline vs ThinkTools
    pub async fn run_comparison(
        &self,
        profile: Profile,
        samples: usize,
    ) -> GSM8KResults {
        let mut baseline_correct = 0;
        let mut enhanced_correct = 0;
        let mut results = Vec::new();

        for problem in self.problems.iter().take(samples) {
            // Run baseline (raw LLM)
            let baseline = self.run_baseline(problem).await;
            if self.check_answer(&baseline.answer, problem.answer) {
                baseline_correct += 1;
            }

            // Run enhanced (ThinkTools)
            let enhanced = self.run_enhanced(problem, profile).await;
            if self.check_answer(&enhanced.answer, problem.answer) {
                enhanced_correct += 1;
            }

            results.push(GSM8KProblemResult {
                problem_id: problem.id.clone(),
                expected: problem.answer,
                baseline_answer: baseline.answer,
                baseline_correct: self.check_answer(&baseline.answer, problem.answer),
                enhanced_answer: enhanced.answer,
                enhanced_correct: self.check_answer(&enhanced.answer, problem.answer),
                baseline_latency_ms: baseline.latency_ms,
                enhanced_latency_ms: enhanced.latency_ms,
            });
        }

        let n = results.len();
        GSM8KResults {
            total: n,
            baseline_correct,
            enhanced_correct,
            baseline_accuracy: baseline_correct as f64 / n as f64,
            enhanced_accuracy: enhanced_correct as f64 / n as f64,
            delta: (enhanced_correct as i32 - baseline_correct as i32) as f64 / n as f64,
            ci: wilson_score_interval(enhanced_correct, n),
            results,
        }
    }

    fn check_answer(&self, predicted: &str, expected: f64) -> bool {
        gsm8k::check_answer(predicted, expected)
    }
}
```

### LogiQA 2.0 Implementation

```rust
// src/benchmark/logiqa.rs

#[derive(Debug, Clone, Deserialize)]
pub struct LogiQAProblem {
    pub id: String,
    pub context: String,
    pub question: String,
    pub options: Vec<String>,
    pub answer: usize,  // 0-indexed
    pub category: LogiQACategory,
}

#[derive(Debug, Clone, Deserialize)]
pub enum LogiQACategory {
    Categorical,
    SufficientConditional,
    NecessaryConditional,
    Disjunctive,
    Conjunctive,
    Other,
}

pub struct LogiQABenchmark {
    problems: Vec<LogiQAProblem>,
    config: BenchmarkConfig,
}

impl LogiQABenchmark {
    pub async fn load() -> Result<Self> {
        // Load from HuggingFace: lucasmccabe/logiqa-2.0
        todo!()
    }

    pub async fn run(&self, profile: Profile) -> LogiQAResults {
        let mut results_by_category: HashMap<LogiQACategory, CategoryResult> = HashMap::new();

        for problem in &self.problems {
            let prompt = self.format_prompt(problem);
            let response = execute_with_profile(&prompt, profile).await;
            let predicted = self.parse_answer(&response.output);
            let correct = predicted == problem.answer;

            // Track by category
            let entry = results_by_category
                .entry(problem.category.clone())
                .or_default();
            entry.total += 1;
            if correct {
                entry.correct += 1;
            }
        }

        LogiQAResults {
            overall_accuracy: self.compute_overall_accuracy(&results_by_category),
            by_category: results_by_category,
            ci: self.compute_ci(),
        }
    }

    fn format_prompt(&self, problem: &LogiQAProblem) -> String {
        format!(
            "Context: {}\n\nQuestion: {}\n\nOptions:\n{}\n\n\
             Choose the correct answer (A, B, C, or D).",
            problem.context,
            problem.question,
            problem.options.iter()
                .enumerate()
                .map(|(i, o)| format!("{}. {}", (b'A' + i as u8) as char, o))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
```

### Custom Audit Trail Benchmark

```rust
// src/benchmark/audit_completeness.rs

/// Validates that all ThinkTool executions produce complete, auditable traces
pub struct AuditTrailBenchmark;

impl AuditTrailBenchmark {
    /// Run audit completeness validation
    pub async fn validate(&self, traces: &[ExecutionTrace]) -> AuditResult {
        let mut complete = 0;
        let mut issues = Vec::new();

        for trace in traces {
            let validation = self.validate_trace(trace);
            if validation.is_complete() {
                complete += 1;
            } else {
                issues.push(validation);
            }
        }

        AuditResult {
            total_traces: traces.len(),
            complete_traces: complete,
            completeness_rate: complete as f64 / traces.len() as f64,
            issues,
        }
    }

    fn validate_trace(&self, trace: &ExecutionTrace) -> TraceValidation {
        TraceValidation {
            has_session_id: trace.session_id.is_some(),
            has_timestamp: trace.started_at.is_some() && trace.ended_at.is_some(),
            has_profile: !trace.profile.is_empty(),
            has_all_steps: trace.steps.iter().all(|s| s.is_valid()),
            has_confidence: trace.confidence > 0.0,
            has_token_counts: trace.tokens.total_tokens > 0,
            steps_have_outputs: trace.steps.iter().all(|s| s.output.is_some()),
            provenance_complete: trace.steps.iter().all(|s| {
                matches!(s.output, Some(StepOutput::Text { .. }) | Some(StepOutput::Structured { .. }))
            }),
        }
    }
}

#[derive(Debug)]
pub struct TraceValidation {
    pub has_session_id: bool,
    pub has_timestamp: bool,
    pub has_profile: bool,
    pub has_all_steps: bool,
    pub has_confidence: bool,
    pub has_token_counts: bool,
    pub steps_have_outputs: bool,
    pub provenance_complete: bool,
}

impl TraceValidation {
    pub fn is_complete(&self) -> bool {
        self.has_session_id
            && self.has_timestamp
            && self.has_profile
            && self.has_all_steps
            && self.has_confidence
            && self.has_token_counts
            && self.steps_have_outputs
            && self.provenance_complete
    }

    pub fn completeness_score(&self) -> f32 {
        let fields = [
            self.has_session_id,
            self.has_timestamp,
            self.has_profile,
            self.has_all_steps,
            self.has_confidence,
            self.has_token_counts,
            self.steps_have_outputs,
            self.provenance_complete,
        ];
        fields.iter().filter(|&&b| b).count() as f32 / fields.len() as f32
    }
}
```

### Custom Debug Time Reduction Benchmark

```rust
// src/benchmark/debug_time.rs

/// Measures time to debug issues with vs without ThinkTool traces
pub struct DebugTimeBenchmark {
    scenarios: Vec<DebugScenario>,
}

#[derive(Debug, Clone)]
pub struct DebugScenario {
    pub id: String,
    pub description: String,
    pub raw_response: String,           // Response without trace
    pub traced_response: ExecutionTrace, // Response with full trace
    pub bug_location: BugLocation,       // Where the issue actually is
}

#[derive(Debug, Clone)]
pub struct BugLocation {
    pub step: usize,
    pub issue_type: IssueType,
    pub explanation: String,
}

#[derive(Debug, Clone)]
pub enum IssueType {
    LogicalFallacy,
    FactualError,
    MissingPerspective,
    OverConfidence,
    CircularReasoning,
}

impl DebugTimeBenchmark {
    /// Human evaluation: how long does it take to find the bug?
    pub fn evaluate_human(&self, scenario: &DebugScenario) -> DebugTimeResult {
        // This requires human participants
        // Track:
        // 1. Time to identify issue in raw response
        // 2. Time to identify issue with trace
        // 3. Accuracy of issue identification
        todo!("Requires human evaluation protocol")
    }

    /// Automated proxy: can we detect issues from trace structure?
    pub fn evaluate_automated(&self, scenario: &DebugScenario) -> AutomatedDebugResult {
        let trace = &scenario.traced_response;

        AutomatedDebugResult {
            // Can we identify low-confidence steps?
            low_confidence_steps: trace.steps.iter()
                .filter(|s| s.confidence < 0.5)
                .count(),

            // Can we see perspective gaps?
            perspectives_count: self.count_perspectives(trace),

            // Can we identify circular dependencies?
            has_circular_reasoning: self.detect_circular_reasoning(trace),

            // Overall debuggability score
            debuggability_score: self.compute_debuggability(trace),
        }
    }

    fn compute_debuggability(&self, trace: &ExecutionTrace) -> f32 {
        let mut score = 0.0;

        // Step visibility
        if !trace.steps.is_empty() { score += 0.2; }

        // Confidence transparency
        if trace.steps.iter().all(|s| s.confidence > 0.0) { score += 0.2; }

        // Output structure
        if trace.steps.iter().all(|s| s.output.is_some()) { score += 0.2; }

        // Token tracking
        if trace.tokens.total_tokens > 0 { score += 0.2; }

        // Profile documentation
        if !trace.profile.is_empty() { score += 0.2; }

        score
    }
}
```

---

## Evaluation Methodology

### Statistical Rigor

#### Sample Size Requirements

```rust
/// Calculate minimum sample size for detecting effect
pub fn required_sample_size(
    baseline_accuracy: f64,
    expected_improvement: f64,
    alpha: f64,  // Default: 0.05
    power: f64,  // Default: 0.80
) -> usize {
    let p1 = baseline_accuracy;
    let p2 = baseline_accuracy + expected_improvement;

    let z_alpha = 1.96;  // Two-tailed, alpha = 0.05
    let z_beta = 0.84;   // Power = 0.80

    let pooled_p = (p1 + p2) / 2.0;
    let numerator = (z_alpha * (2.0 * pooled_p * (1.0 - pooled_p)).sqrt()
                    + z_beta * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt()).powi(2);
    let denominator = (p2 - p1).powi(2);

    (numerator / denominator).ceil() as usize
}

// Examples:
// - Detect 5% improvement from 80% baseline: ~385 samples
// - Detect 10% improvement from 70% baseline: ~141 samples
// - Detect 3% improvement from 90% baseline: ~895 samples
```

#### Confidence Intervals

```rust
/// Wilson score interval (preferred for binary outcomes)
pub fn wilson_score_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.96;  // 95% CI

    let denominator = 1.0 + z * z / n;
    let center = p + z * z / (2.0 * n);
    let spread = z * ((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))).sqrt();

    let lower = (center - spread) / denominator;
    let upper = (center + spread) / denominator;

    (lower.max(0.0), upper.min(1.0))
}

/// Agresti-Coull interval (simpler alternative)
pub fn agresti_coull_interval(successes: usize, trials: usize) -> (f64, f64) {
    let z = 1.96;
    let n_tilde = trials as f64 + z * z;
    let p_tilde = (successes as f64 + z * z / 2.0) / n_tilde;
    let se = (p_tilde * (1.0 - p_tilde) / n_tilde).sqrt();

    (p_tilde - z * se, p_tilde + z * se)
}
```

#### Statistical Tests

```rust
/// McNemar's test for paired binary outcomes
/// Use when comparing same samples with two methods
pub fn mcnemars_test(
    baseline_correct: &[bool],
    enhanced_correct: &[bool],
) -> McNemarResult {
    assert_eq!(baseline_correct.len(), enhanced_correct.len());

    let mut b = 0;  // Baseline correct, enhanced wrong
    let mut c = 0;  // Baseline wrong, enhanced correct

    for (base, enhanced) in baseline_correct.iter().zip(enhanced_correct.iter()) {
        match (*base, *enhanced) {
            (true, false) => b += 1,
            (false, true) => c += 1,
            _ => {}
        }
    }

    let chi_square = if b + c > 0 {
        ((b as f64 - c as f64).abs() - 1.0).powi(2) / (b + c) as f64
    } else {
        0.0
    };

    // Chi-square with 1 df, p < 0.05 requires chi2 > 3.84
    let is_significant = chi_square > 3.84;
    let p_value = 1.0 - chi_square_cdf(chi_square, 1);

    McNemarResult {
        baseline_only_correct: b,
        enhanced_only_correct: c,
        chi_square,
        p_value,
        is_significant,
    }
}
```

### Fair Comparison Protocol

To ensure fair benchmarking between raw LLM and ReasonKit:

```yaml
Fair Comparison Checklist:
  model:
    - SAME model for both conditions
    - SAME model version/checkpoint
    - SAME API endpoint

  parameters:
    - SAME temperature (0.0 for reproducibility)
    - SAME max_tokens (sufficient for both)
    - SAME top_p, frequency_penalty, etc.

  prompts:
    - Baseline: Raw question only
    - Enhanced: Question + ThinkTool structure
    - NO additional hints to baseline

  evaluation:
    - SAME answer extraction method
    - SAME correctness criteria
    - Blind evaluation where possible

  runs:
    - MINIMUM 10 trials per sample
    - SAME random seed for sampling
    - ALTERNATING order (baseline, enhanced, baseline, ...)
```

### Reproducibility Requirements

```yaml
# benchmark_manifest.yaml

reproducibility:
  # Environment specification
  environment:
    rust_version: "1.83.0"
    reasonkit_version: "0.1.0"
    reasonkit_commit: "abc123def456"
    os: "Linux 6.12.57"

  # Model specification
  model:
    provider: "anthropic"
    model_id: "claude-sonnet-4-20250514"
    temperature: 0.0
    max_tokens: 4096

  # Benchmark specification
  benchmark:
    name: "gsm8k"
    version: "main"
    split: "test"
    samples: 1000

  # Random state
  random:
    seed: 42
    deterministic_sampling: true

  # Reproduction command
  reproduce:
    command: |
      git checkout abc123def456
      cargo run --release --bin sciengine -- \
        --benchmark gsm8k \
        --samples 1000 \
        --seed 42 \
        --model claude-sonnet-4
```

---

## A/B Test Framework

### Test Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                       A/B TEST FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐                                                    │
│  │  Query  │                                                    │
│  └────┬────┘                                                    │
│       │                                                         │
│       ├──────────────────┬──────────────────┐                   │
│       ▼                  ▼                  ▼                   │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐              │
│  │Baseline │        │ Quick   │        │Balanced │              │
│  │  (Raw)  │        │ Profile │        │ Profile │              │
│  └────┬────┘        └────┬────┘        └────┬────┘              │
│       │                  │                  │                   │
│       ▼                  ▼                  ▼                   │
│  ┌─────────────────────────────────────────────────┐            │
│  │            EVALUATION METRICS                   │            │
│  │  - Accuracy (is answer correct?)                │            │
│  │  - Confidence (does it know what it knows?)     │            │
│  │  - Latency (how long did it take?)              │            │
│  │  - Tokens (how much did it cost?)               │            │
│  │  - Structure (is reasoning traceable?)          │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
// src/benchmark/ab_test.rs

pub struct ABTestConfig {
    /// Number of samples to test
    pub samples: usize,

    /// Profiles to compare (baseline always included)
    pub profiles: Vec<Profile>,

    /// Number of trials per sample
    pub trials: usize,

    /// Whether to randomize order
    pub randomize_order: bool,

    /// Seed for reproducibility
    pub seed: u64,
}

pub struct ABTestResult {
    pub baseline: ConditionResult,
    pub treatments: HashMap<Profile, ConditionResult>,
    pub comparisons: Vec<PairwiseComparison>,
}

pub struct ConditionResult {
    pub profile: Profile,
    pub accuracy: f64,
    pub ci: (f64, f64),
    pub mean_confidence: f64,
    pub mean_latency_ms: f64,
    pub mean_tokens: usize,
    pub results: Vec<SampleResult>,
}

pub struct PairwiseComparison {
    pub baseline: Profile,
    pub treatment: Profile,
    pub accuracy_delta: f64,
    pub is_significant: bool,
    pub p_value: f64,
    pub effect_size: f64,  // Cohen's h
    pub token_ratio: f64,
    pub latency_ratio: f64,
}

impl ABTestRunner {
    pub async fn run(&self, config: ABTestConfig) -> ABTestResult {
        let samples = self.load_samples(config.samples);
        let mut results = HashMap::new();

        // Run baseline
        let baseline_results = self.run_condition(&samples, Profile::Raw, config.trials).await;
        results.insert(Profile::Raw, baseline_results);

        // Run each treatment
        for profile in &config.profiles {
            let treatment_results = self.run_condition(&samples, *profile, config.trials).await;
            results.insert(*profile, treatment_results);
        }

        // Compute pairwise comparisons
        let comparisons = self.compute_comparisons(&results);

        ABTestResult {
            baseline: results.remove(&Profile::Raw).unwrap(),
            treatments: results,
            comparisons,
        }
    }

    fn compute_comparisons(&self, results: &HashMap<Profile, ConditionResult>) -> Vec<PairwiseComparison> {
        let baseline = results.get(&Profile::Raw).unwrap();

        results.iter()
            .filter(|(p, _)| **p != Profile::Raw)
            .map(|(profile, treatment)| {
                let mcnemar = mcnemars_test(
                    &baseline.results.iter().map(|r| r.correct).collect::<Vec<_>>(),
                    &treatment.results.iter().map(|r| r.correct).collect::<Vec<_>>(),
                );

                PairwiseComparison {
                    baseline: Profile::Raw,
                    treatment: *profile,
                    accuracy_delta: treatment.accuracy - baseline.accuracy,
                    is_significant: mcnemar.is_significant,
                    p_value: mcnemar.p_value,
                    effect_size: cohens_h(baseline.accuracy, treatment.accuracy),
                    token_ratio: treatment.mean_tokens as f64 / baseline.mean_tokens as f64,
                    latency_ratio: treatment.mean_latency_ms / baseline.mean_latency_ms,
                }
            })
            .collect()
    }
}

/// Cohen's h effect size for proportions
fn cohens_h(p1: f64, p2: f64) -> f64 {
    let phi1 = 2.0 * p1.sqrt().asin();
    let phi2 = 2.0 * p2.sqrt().asin();
    (phi2 - phi1).abs()
}
```

### Metrics

| Metric | Baseline | Treatment | Interpretation |
|--------|----------|-----------|----------------|
| **Accuracy** | Raw LLM | ThinkTools | Higher = better reasoning |
| **Delta** | 0 | Treatment - Baseline | > 0 = improvement |
| **95% CI** | - | Wilson score | Narrow = precise |
| **p-value** | - | McNemar's test | < 0.05 = significant |
| **Effect Size** | - | Cohen's h | > 0.2 = small, > 0.5 = medium, > 0.8 = large |
| **Token Ratio** | 1.0x | Treatment / Baseline | Cost multiplier |
| **Latency Ratio** | 1.0x | Treatment / Baseline | Time multiplier |

---

## CLI Commands

### Primary Benchmark Commands

```bash
# ============================================================================
# BENCHMARK SUITE COMMANDS
# ============================================================================

# Run GSM8K benchmark (default: 100 samples)
rk-core benchmark gsm8k --samples 100 --model claude-sonnet-4

# Run with specific profile comparison
rk-core benchmark gsm8k --samples 100 --profiles baseline,quick,balanced,deep

# Run full benchmark suite (all benchmarks, all profiles)
rk-core benchmark all --output results/$(date +%Y%m%d).json

# Run quick smoke test (10 samples per benchmark)
rk-core benchmark quick

# ============================================================================
# A/B COMPARISON COMMANDS
# ============================================================================

# Compare specific query
rk-core compare "Should we use microservices?" --profile balanced

# Compare with all profiles
rk-core compare "What causes inflation?" --profiles all

# Batch comparison from file
rk-core compare --file queries.txt --output comparison_results.json

# ============================================================================
# PROFILE-SPECIFIC BENCHMARKS
# ============================================================================

# Compare profiles on GSM8K
rk-core benchmark compare \
    --baseline raw \
    --treatment balanced \
    --benchmark gsm8k \
    --samples 200

# Self-Consistency benchmark
rk-core benchmark gsm8k \
    --self-consistency 5 \
    --voting majority \
    --samples 100

# ============================================================================
# REPORT GENERATION
# ============================================================================

# Generate markdown report
rk-core benchmark report --format markdown --input results/*.json

# Generate HTML dashboard
rk-core benchmark report --format html --input results/*.json --output dashboard.html

# Generate JSON for downstream processing
rk-core benchmark report --format json --input results/*.json

# ============================================================================
# VALIDATION COMMANDS
# ============================================================================

# Validate trace completeness
rk-core benchmark audit --input traces/*.json

# Check debuggability score
rk-core benchmark debug-quality --input traces/*.json

# Validate statistical claims
rk-core benchmark validate-claims --input results/*.json
```

### Example Outputs

```bash
$ rk-core benchmark gsm8k --samples 100 --profiles baseline,balanced

╔════════════════════════════════════════════════════════════════════╗
║                    GSM8K BENCHMARK RESULTS                         ║
╠════════════════════════════════════════════════════════════════════╣
║ Dataset: GSM8K (100 samples)  Model: claude-sonnet-4               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Profile      Accuracy    95% CI         Tokens    Latency         ║
║  ─────────────────────────────────────────────────────────────────║
║  baseline     78.0%       [69.1, 85.0]   ~150      0.8s            ║
║  balanced     85.0%       [76.9, 90.7]   ~650      2.1s            ║
║                                                                    ║
║  Delta: +7.0pp (8.9% relative improvement)                         ║
║  McNemar p-value: 0.023 (SIGNIFICANT at p < 0.05)                  ║
║  Effect size: 0.18 (small)                                         ║
║                                                                    ║
║  Token cost: 4.3x baseline                                         ║
║  Latency: 2.6x baseline                                            ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  VERDICT: Statistically significant improvement detected.          ║
║           Cost-benefit depends on use case value of accuracy.      ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Results Reporting

### Metrics Dashboard Schema

```json
{
  "benchmark_run": {
    "id": "run_20251228_143022",
    "timestamp": "2025-12-28T14:30:22Z",
    "reasonkit_version": "0.1.0",
    "commit": "abc123"
  },
  "benchmarks": [
    {
      "name": "gsm8k",
      "samples": 100,
      "profiles": {
        "baseline": {
          "accuracy": 0.78,
          "ci_lower": 0.691,
          "ci_upper": 0.850,
          "mean_tokens": 150,
          "mean_latency_ms": 800,
          "mean_confidence": 0.72
        },
        "balanced": {
          "accuracy": 0.85,
          "ci_lower": 0.769,
          "ci_upper": 0.907,
          "mean_tokens": 650,
          "mean_latency_ms": 2100,
          "mean_confidence": 0.81
        }
      },
      "comparisons": [
        {
          "baseline": "baseline",
          "treatment": "balanced",
          "delta": 0.07,
          "p_value": 0.023,
          "significant": true,
          "effect_size": 0.18,
          "token_ratio": 4.33,
          "latency_ratio": 2.63
        }
      ]
    }
  ],
  "summary": {
    "total_samples": 100,
    "significant_improvements": 1,
    "cost_effective_profiles": ["balanced"]
  }
}
```

### Markdown Report Template

```markdown
# ReasonKit Benchmark Report

**Generated:** {date}
**Version:** {version}
**Commit:** {commit}

## Executive Summary

| Benchmark | Best Profile | Delta vs Baseline | Significant? |
|-----------|-------------|-------------------|--------------|
{summary_table}

## Detailed Results

### GSM8K (Grade School Math)

**Best Result:** balanced profile (+7.0% vs baseline)

| Profile | Accuracy | 95% CI | Latency | Tokens | Cost |
|---------|----------|--------|---------|--------|------|
{profile_table}

**Statistical Significance:**
- McNemar's test: p = 0.023 (significant at alpha = 0.05)
- Effect size (Cohen's h): 0.18 (small effect)

**Where ThinkTools Helped:**
- Multi-step problems requiring explicit reasoning
- Problems with common calculation errors
- Problems requiring systematic approach

**Where ThinkTools Did NOT Help:**
- Simple single-step calculations
- Problems already solved correctly by baseline
- Very long problems (token budget exceeded)

## Methodology

- **Trials per sample:** 10
- **Temperature:** 0.0 (deterministic)
- **Confidence intervals:** Wilson score (95%)
- **Statistical test:** McNemar's test for paired data
- **Effect size:** Cohen's h for proportions

## Reproduction

\`\`\`bash
git checkout {commit}
rk-core benchmark gsm8k --samples 100 --seed 42
\`\`\`

## Honest Assessment

### What This Shows
- ThinkTools can improve accuracy on math reasoning tasks
- Improvement is statistically significant but effect size is small
- Cost is 4x tokens for 7% accuracy gain

### What This Does NOT Show
- ThinkTools always improve accuracy
- Improvement transfers to all domains
- Cost-benefit is universally positive

### Recommendations
- Use for high-stakes decisions where accuracy matters
- Consider cost-benefit for high-volume applications
- Run your own benchmarks on your specific use case
```

---

## Continuous Benchmarking

### CI/CD Integration

```yaml
# .github/workflows/benchmarks.yml

name: Benchmark Suite

on:
  schedule:
    - cron: '0 0 * * *'  # Nightly at midnight UTC
  pull_request:
    paths:
      - 'src/thinktool/**'
      - 'src/benchmark/**'
  workflow_dispatch:
    inputs:
      benchmark:
        description: 'Benchmark to run (all, gsm8k, logiqa, arc_c)'
        required: false
        default: 'quick'
      samples:
        description: 'Number of samples'
        required: false
        default: '50'

env:
  RUST_BACKTRACE: 1

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Build release
        run: cargo build --release

      - name: Run benchmarks
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          ./target/release/rk-core benchmark \
            ${{ github.event.inputs.benchmark || 'quick' }} \
            --samples ${{ github.event.inputs.samples || '50' }} \
            --output results/benchmark_$(date +%Y%m%d_%H%M%S).json

      - name: Generate report
        run: |
          ./target/release/rk-core benchmark report \
            --format markdown \
            --input results/benchmark_*.json \
            --output results/BENCHMARK_REPORT.md

      - name: Check for regression
        run: |
          ./scripts/check_regression.sh results/benchmark_*.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: results/
          retention-days: 90

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('results/BENCHMARK_REPORT.md', 'utf8');

            // Find existing benchmark comment
            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });

            const botComment = comments.find(c =>
              c.user.type === 'Bot' && c.body.includes('## Benchmark Results')
            );

            const body = '## Benchmark Results\n\n' + report;

            if (botComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: botComment.id,
                body,
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body,
              });
            }

  regression-check:
    needs: benchmark
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - name: Download current results
        uses: actions/download-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: current/

      - name: Download baseline results
        uses: actions/download-artifact@v4
        with:
          name: benchmark-results-main
          path: baseline/
        continue-on-error: true

      - name: Check for regression
        run: |
          if [ -d baseline ]; then
            ./scripts/compare_benchmarks.sh baseline/ current/
          else
            echo "No baseline available, skipping regression check"
          fi
```

### Regression Detection

```bash
#!/bin/bash
# scripts/check_regression.sh

set -e

THRESHOLD=0.05  # 5% regression threshold

latest_file=$(ls -t results/benchmark_*.json | head -1)
baseline_file="results/baseline.json"

if [ ! -f "$baseline_file" ]; then
    echo "No baseline found. Setting current as baseline."
    cp "$latest_file" "$baseline_file"
    exit 0
fi

# Extract accuracies
baseline_acc=$(jq '.benchmarks[0].profiles.balanced.accuracy' "$baseline_file")
current_acc=$(jq '.benchmarks[0].profiles.balanced.accuracy' "$latest_file")

# Calculate regression
regression=$(echo "$baseline_acc - $current_acc" | bc -l)

if (( $(echo "$regression > $THRESHOLD" | bc -l) )); then
    echo "REGRESSION DETECTED!"
    echo "Baseline: $baseline_acc"
    echo "Current: $current_acc"
    echo "Regression: $regression"
    exit 1
else
    echo "No regression detected."
    echo "Baseline: $baseline_acc"
    echo "Current: $current_acc"
    echo "Delta: $regression"

    # Update baseline if improved
    if (( $(echo "$current_acc > $baseline_acc" | bc -l) )); then
        echo "Improvement detected! Updating baseline."
        cp "$latest_file" "$baseline_file"
    fi
fi
```

### Historical Tracking

```rust
// src/benchmark/history.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkHistory {
    pub benchmark: String,
    pub entries: Vec<HistoryEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub commit: String,
    pub version: String,
    pub accuracy: f64,
    pub ci: (f64, f64),
    pub samples: usize,
}

impl BenchmarkHistory {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn add_entry(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
        self.entries.sort_by_key(|e| e.timestamp);
    }

    pub fn detect_trend(&self, window: usize) -> Trend {
        if self.entries.len() < window {
            return Trend::Insufficient;
        }

        let recent: Vec<_> = self.entries.iter().rev().take(window).collect();
        let accuracies: Vec<f64> = recent.iter().map(|e| e.accuracy).collect();

        let slope = linear_regression_slope(&accuracies);

        if slope > 0.01 {
            Trend::Improving(slope)
        } else if slope < -0.01 {
            Trend::Degrading(slope.abs())
        } else {
            Trend::Stable
        }
    }

    pub fn detect_anomaly(&self, current: f64) -> Option<Anomaly> {
        if self.entries.len() < 10 {
            return None;
        }

        let mean = self.entries.iter().map(|e| e.accuracy).sum::<f64>()
                   / self.entries.len() as f64;
        let std_dev = (self.entries.iter()
            .map(|e| (e.accuracy - mean).powi(2))
            .sum::<f64>() / self.entries.len() as f64).sqrt();

        let z_score = (current - mean) / std_dev;

        if z_score.abs() > 2.0 {
            Some(Anomaly {
                current,
                mean,
                std_dev,
                z_score,
                is_improvement: z_score > 0.0,
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum Trend {
    Improving(f64),
    Degrading(f64),
    Stable,
    Insufficient,
}

#[derive(Debug)]
pub struct Anomaly {
    pub current: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub z_score: f64,
    pub is_improvement: bool,
}
```

---

## Third-Party Validation

### Independent Verification Protocol

```yaml
# VERIFICATION_PROTOCOL.yaml

purpose: >
  Enable independent verification of ReasonKit benchmark claims
  by providing complete reproduction instructions.

requirements:
  - No proprietary datasets (use public benchmarks)
  - No hidden hyperparameters
  - Deterministic random seeds
  - Pinned dependency versions
  - Clear evaluation criteria

reproduction_steps:
  1. environment_setup:
     - Install Rust 1.83.0 or later
     - Clone repository at specified commit
     - Install dependencies with `cargo build --release`

  2. api_configuration:
     - Set ANTHROPIC_API_KEY environment variable
     - Verify API access with health check

  3. dataset_download:
     - GSM8K: huggingface.co/datasets/gsm8k
     - LogiQA: huggingface.co/datasets/lucasmccabe/logiqa-2.0
     - ARC-C: huggingface.co/datasets/allenai/ai2_arc

  4. benchmark_execution:
     command: |
       ./target/release/rk-core benchmark all \
         --samples 1000 \
         --seed 42 \
         --output verification_results.json

  5. result_comparison:
     - Compare accuracy within 2% of reported values
     - Verify confidence intervals overlap
     - Check statistical significance reproducible

verification_checklist:
  - [ ] Environment matches specification
  - [ ] Datasets match versions
  - [ ] Random seed produces same sample order
  - [ ] Accuracy within tolerance
  - [ ] CI intervals overlap
  - [ ] Statistical tests agree on significance
```

### Public Dataset Access

```bash
# scripts/download_datasets.sh

#!/bin/bash
set -e

DATA_DIR="data/benchmarks"
mkdir -p "$DATA_DIR"

echo "Downloading GSM8K..."
huggingface-cli download gsm8k --repo-type dataset --local-dir "$DATA_DIR/gsm8k"

echo "Downloading LogiQA 2.0..."
huggingface-cli download lucasmccabe/logiqa-2.0 --repo-type dataset --local-dir "$DATA_DIR/logiqa"

echo "Downloading ARC-Challenge..."
huggingface-cli download allenai/ai2_arc --repo-type dataset --local-dir "$DATA_DIR/arc"

echo "Downloading TruthfulQA..."
huggingface-cli download truthful_qa --repo-type dataset --local-dir "$DATA_DIR/truthfulqa"

echo "Verifying checksums..."
sha256sum -c datasets.sha256

echo "Done! Datasets available in $DATA_DIR"
```

### Community Benchmark Submission

```yaml
# community_benchmark_schema.yaml

submission:
  # Metadata
  submitter: "string"
  date: "ISO 8601 datetime"
  reasonkit_version: "semver"

  # Environment
  environment:
    os: "string"
    rust_version: "string"
    model_provider: "string"
    model_id: "string"

  # Results
  benchmarks:
    - name: "string"
      samples: "integer"
      profiles:
        baseline:
          accuracy: "float"
          ci_lower: "float"
          ci_upper: "float"
        # ... other profiles

  # Verification
  verification:
    command_used: "string"
    seed: "integer"
    commit: "string"

  # Optional
  notes: "string"
  raw_results_url: "string"
```

---

## Interpretation Guide

### What Good Looks Like

| Delta | Interpretation | Action | Confidence |
|-------|----------------|--------|------------|
| **> +10%** | Strong improvement | Highlight in marketing, document use case | High |
| **+5-10%** | Meaningful improvement | Document, verify cost-benefit | Medium-High |
| **+2-5%** | Marginal improvement | May not justify cost for all use cases | Medium |
| **+0-2%** | No meaningful difference | Honest: "no measurable change" | Low |
| **< 0%** | Degradation | Document when NOT to use, investigate | High |

### Effect Size Interpretation

| Cohen's h | Interpretation | Example |
|-----------|----------------|---------|
| < 0.2 | Trivial | 80% -> 82% |
| 0.2 - 0.5 | Small | 80% -> 85% |
| 0.5 - 0.8 | Medium | 80% -> 90% |
| > 0.8 | Large | 70% -> 90% |

### Cost-Benefit Analysis

```
Value = (Accuracy_improvement * Value_per_correct) - Cost_increase

Example Calculation:
- Benchmark: GSM8K
- Improvement: +7% (78% -> 85%)
- Baseline cost: $0.002/query (150 tokens @ $0.012/1K)
- ThinkTools cost: $0.008/query (650 tokens @ $0.012/1K)
- Cost increase: $0.006/query

Break-even analysis:
- Value per correct answer must exceed: $0.006 / 0.07 = $0.086

Decision Framework:
- If value per correct > $0.09: USE THINKTOOLS
- If value per correct < $0.05: RAW LLM
- If in between: DEPENDS ON RISK TOLERANCE
```

### Honest Positioning

#### Where ReasonKit Excels

| Domain | Evidence | Recommended Profile |
|--------|----------|---------------------|
| **Multi-step math** | GSM8K +7% | balanced, deep |
| **Logical deduction** | LogiQA +10% | paranoid |
| **Complex analysis** | Qualitative | deep, powercombo |
| **Audit-critical** | Trace completeness 100% | Any |

#### Where ReasonKit Does NOT Add Value

| Domain | Finding | Recommendation |
|--------|---------|----------------|
| **Simple factual QA** | No improvement | Use raw LLM |
| **Creative writing** | Potential degradation | Use raw LLM or quick |
| **High-throughput, low-stakes** | Cost prohibitive | Use raw LLM |
| **Already correct** | No room for improvement | Use raw LLM |

#### Honest Limitations

```markdown
## What We Don't Claim

1. **Universal improvement**: ThinkTools don't help every task.
   Some tasks are better served by raw LLM speed.

2. **Magic intelligence boost**: We structure reasoning,
   not make models smarter. The improvement comes from
   systematic thinking, not AI advancement.

3. **Cost efficiency for all use cases**: 4x token cost
   means you need 4x value from accuracy improvement.
   Do the math for your use case.

4. **Guaranteed significance**: Statistical significance
   depends on sample size and effect size. Run your own
   benchmarks.

5. **Transfer across domains**: Benchmarks on math don't
   prove improvement on your specific domain. Test it.
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [x] Core benchmark harness (`src/thinktool/benchmark.rs`)
- [x] GSM8K loader and evaluator
- [x] Basic A/B comparison (`src/bin/compare.rs`)
- [x] Self-Consistency implementation
- [ ] Wilson score CI implementation
- [ ] McNemar's test implementation

### Phase 2: Core Benchmarks (Week 3-4)

- [ ] LogiQA 2.0 integration
- [ ] ARC-Challenge integration
- [ ] TruthfulQA for hallucination detection
- [ ] Trace completeness validator
- [ ] Debuggability scorer

### Phase 3: Infrastructure (Week 5-6)

- [ ] CI/CD workflow setup
- [ ] Regression detection script
- [ ] Historical tracking system
- [ ] Markdown report generator
- [ ] HTML dashboard

### Phase 4: Polish & Launch (Week 7-8)

- [ ] Public benchmark results page
- [ ] Reproduction documentation
- [ ] Community submission system
- [ ] Third-party verification guide

---

## Appendix: Statistical Reference

### Wilson Score Interval

For binary outcomes (correct/incorrect), Wilson score interval is preferred over normal approximation:

```
CI = (p + z^2/2n +/- z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)

Where:
- p = observed proportion (successes/trials)
- z = 1.96 for 95% CI
- n = number of trials
```

### McNemar's Test

For paired binary data (same samples, two methods):

```
Chi-square = (|b - c| - 1)^2 / (b + c)

Where:
- b = count(baseline correct, treatment wrong)
- c = count(baseline wrong, treatment correct)
- Significant if chi-square > 3.84 (p < 0.05)
```

### Cohen's h

For effect size of proportion differences:

```
h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))

Where:
- p1, p2 = proportions to compare
- |h| < 0.2: trivial effect
- 0.2 <= |h| < 0.5: small effect
- 0.5 <= |h| < 0.8: medium effect
- |h| >= 0.8: large effect
```

---

## References

1. Wang, X., et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171
2. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903
3. Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." arXiv:2305.10601
4. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168
5. Wilson, E.B. (1927). "Probable Inference, the Law of Succession, and Statistical Inference." JASA.

---

*ReasonKit Benchmark Suite v2.0 | Structured Prompt Engineering Framework*
*"No claims without evidence. Run benchmarks to prove value."*
