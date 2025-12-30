# SciEngine Benchmark Suite Design

> **Purpose:** Reproducible, statistically rigorous evaluation of ReasonKit ThinkTools
> **Version:** 1.0 | **Status:** Design Complete
> **Philosophy:** No claims without evidence. Run benchmarks to prove value.

---

## Design Principles

### 1. Statistical Rigor

- Report 95% confidence intervals on ALL results
- Minimum 10 trials per configuration
- Model responses as Bernoulli trials
- Use temperature=0 for reproducibility

### 2. Reproducibility

- All prompts publicly available
- Document: model version, GPU type, batch size, precision
- requirements.txt with exact versions
- Seed all random operations

### 3. Honest Reporting

- Show where ThinkTools help AND hurt
- No cherry-picking favorable results
- Include cost/latency alongside accuracy

---

## Benchmark Selection

### Primary Benchmarks (Run These)

| Benchmark          | Why                                  | ReasonKit Opportunity       |
| ------------------ | ------------------------------------ | --------------------------- |
| **LogiQA 2.0**     | 27.8pp gap between GPT-4 and QwQ-32B | Test ProofGuard, LaserLogic |
| **MMLU-Pro**       | 10-choice, not saturated             | Knowledge + reasoning       |
| **BIG-Bench Hard** | 23 challenging tasks                 | Test GigaThink, BedRock     |
| **Game of 24**     | 74% vs 4% ToT demonstration          | Marketing-friendly demo     |

### Avoid (Saturated/Problematic)

| Benchmark | Problem                                        |
| --------- | ---------------------------------------------- |
| GSM8K     | Saturated (GPT-4 90%+), contamination concerns |
| MMLU      | 6.5% question error rate, saturated            |
| HellaSwag | Near-ceiling performance                       |

### Where Structured Reasoning Helps

| Domain              | Improvement Evidence     |
| ------------------- | ------------------------ |
| Multi-step math     | GSM8K, MATH, AIME        |
| Logical reasoning   | LogiQA, ReClor           |
| Algorithmic puzzles | Game of 24, code puzzles |
| Complex QA          | HotpotQA, multi-hop      |

### Where Structured Reasoning Hurts

| Domain               | Issue                           |
| -------------------- | ------------------------------- |
| Clinical text        | 86.3% of LLMs degraded with CoT |
| Simple tasks         | Overthinking penalty            |
| Small models (<100B) | Illogical reasoning chains      |

---

## Architecture

```
reasonkit-core/
├── benches/
│   └── sciengine/
│       ├── mod.rs                 # Benchmark orchestrator
│       ├── logiqa.rs              # LogiQA 2.0 evaluation
│       ├── mmlu_pro.rs            # MMLU-Pro evaluation
│       ├── big_bench_hard.rs      # BBH evaluation
│       ├── game_of_24.rs          # Game of 24 (marketing demo)
│       ├── comparison.rs          # A/B: raw vs ThinkTools
│       └── statistical.rs         # CI calculation, trials
│
├── data/
│   └── benchmarks/
│       ├── logiqa/                # Dataset cache
│       ├── mmlu_pro/
│       ├── big_bench_hard/
│       └── game_of_24/
│
└── scripts/
    └── benchmark/
        ├── run_full_suite.sh      # All benchmarks
        ├── run_quick.sh           # Smoke test (10 samples)
        └── report_generator.py    # Markdown/HTML reports
```

---

## Implementation

### Core Benchmark Runner

```rust
// benches/sciengine/mod.rs

use reasonkit_core::thinktool::profiles::Profile;

pub struct BenchmarkConfig {
    /// Number of trials per sample
    pub trials: usize,            // Default: 10
    /// Temperature for LLM calls
    pub temperature: f32,         // Default: 0.0
    /// Maximum samples per benchmark
    pub max_samples: Option<usize>,
    /// Which profiles to test
    pub profiles: Vec<Profile>,
    /// Model to benchmark
    pub model: String,
}

pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub profile: Profile,
    pub accuracy: f64,
    pub ci_lower: f64,           // 95% CI
    pub ci_upper: f64,
    pub trials: usize,
    pub samples: usize,
    pub mean_latency_ms: f64,
    pub mean_tokens: usize,
    pub cost_usd: f64,
}

impl BenchmarkConfig {
    pub fn default() -> Self {
        Self {
            trials: 10,
            temperature: 0.0,
            max_samples: None,
            profiles: vec![
                Profile::Quick,
                Profile::Balanced,
                Profile::Deep,
                Profile::Paranoid,
            ],
            model: "claude-sonnet-4".to_string(),
        }
    }
}
```

### Statistical Analysis

```rust
// benches/sciengine/statistical.rs

/// Calculate 95% confidence interval for accuracy
/// Using Wilson score interval (better for small samples)
pub fn wilson_score_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.96; // 95% CI

    let denominator = 1.0 + z * z / n;
    let center = p + z * z / (2.0 * n);
    let spread = z * ((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))).sqrt();

    let lower = (center - spread) / denominator;
    let upper = (center + spread) / denominator;

    (lower.max(0.0), upper.min(1.0))
}

/// Run multiple trials and aggregate results
pub fn run_trials<F>(
    sample_fn: F,
    n_trials: usize,
) -> TrialResult
where
    F: Fn() -> bool
{
    let mut successes = 0;
    let mut latencies = Vec::with_capacity(n_trials);

    for _ in 0..n_trials {
        let start = Instant::now();
        if sample_fn() {
            successes += 1;
        }
        latencies.push(start.elapsed().as_millis());
    }

    let (ci_lower, ci_upper) = wilson_score_interval(successes, n_trials);

    TrialResult {
        successes,
        trials: n_trials,
        accuracy: successes as f64 / n_trials as f64,
        ci_lower,
        ci_upper,
        mean_latency_ms: latencies.iter().sum::<u128>() as f64 / n_trials as f64,
    }
}
```

### LogiQA 2.0 Benchmark

```rust
// benches/sciengine/logiqa.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct LogiQASample {
    pub id: String,
    pub context: String,
    pub question: String,
    pub options: Vec<String>,
    pub answer: usize,  // 0-indexed correct answer
    pub reasoning_type: String,  // categorical, matching, etc.
}

pub struct LogiQABenchmark {
    samples: Vec<LogiQASample>,
    config: BenchmarkConfig,
}

impl LogiQABenchmark {
    pub async fn load() -> Result<Self> {
        // Load from HuggingFace datasets
        // https://huggingface.co/datasets/lucasmccabe/logiqa-2.0
        todo!()
    }

    pub async fn run(&self, profile: Profile) -> BenchmarkResult {
        let mut correct = 0;
        let mut total = 0;
        let mut latencies = Vec::new();

        for sample in &self.samples {
            let prompt = format!(
                "Context: {}\n\nQuestion: {}\n\nOptions:\n{}\n\nAnswer with just the letter.",
                sample.context,
                sample.question,
                sample.options.iter()
                    .enumerate()
                    .map(|(i, o)| format!("{}. {}", (b'A' + i as u8) as char, o))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            let start = Instant::now();
            let result = execute_with_profile(&prompt, profile).await;
            latencies.push(start.elapsed());

            if parse_answer(&result.output) == sample.answer {
                correct += 1;
            }
            total += 1;
        }

        let (ci_lower, ci_upper) = wilson_score_interval(correct, total);

        BenchmarkResult {
            benchmark_name: "LogiQA 2.0".to_string(),
            profile,
            accuracy: correct as f64 / total as f64,
            ci_lower,
            ci_upper,
            trials: total,
            samples: self.samples.len(),
            mean_latency_ms: latencies.iter().map(|d| d.as_millis()).sum::<u128>() as f64 / total as f64,
            mean_tokens: 0, // TODO: track
            cost_usd: 0.0,  // TODO: track
        }
    }
}
```

### Game of 24 (Marketing Demo)

```rust
// benches/sciengine/game_of_24.rs

/// Game of 24: Use 4 numbers and +,-,*,/ to make 24
/// Tree-of-Thoughts showed 74% vs 4% baseline (Yao et al.)
/// This is our marketing demo benchmark.

#[derive(Debug)]
pub struct GameOf24Sample {
    pub numbers: [i32; 4],
    pub solution: Option<String>,  // Pre-computed for validation
}

impl GameOf24Sample {
    pub fn random() -> Self {
        // Generate valid puzzles
        todo!()
    }

    pub fn standard_set() -> Vec<Self> {
        // The exact set from Yao et al. for reproducibility
        vec![
            Self { numbers: [1, 2, 3, 4], solution: Some("(1+2+3)*4".into()) },
            Self { numbers: [2, 3, 4, 5], solution: Some("(2+3+5)*4/2".into()) },
            // ... full set
        ]
    }
}

pub fn validate_solution(numbers: &[i32; 4], expr: &str) -> bool {
    // Parse expression, verify uses all 4 numbers exactly once, equals 24
    todo!()
}

/// Run Game of 24 with different ThinkTool profiles
pub async fn benchmark_game_of_24(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let samples = GameOf24Sample::standard_set();
    let mut results = Vec::new();

    for profile in &config.profiles {
        let mut correct = 0;
        let mut total = 0;

        for sample in &samples {
            for _ in 0..config.trials {
                let prompt = format!(
                    "Use the numbers {} to make 24 using +, -, *, /.\n\
                     Each number must be used exactly once.\n\
                     Output only the expression.",
                    sample.numbers.iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                let result = execute_with_profile(&prompt, *profile).await;

                if validate_solution(&sample.numbers, &result.output) {
                    correct += 1;
                }
                total += 1;
            }
        }

        let (ci_lower, ci_upper) = wilson_score_interval(correct, total);

        results.push(BenchmarkResult {
            benchmark_name: "Game of 24".to_string(),
            profile: *profile,
            accuracy: correct as f64 / total as f64,
            ci_lower,
            ci_upper,
            trials: total,
            samples: samples.len(),
            mean_latency_ms: 0.0,
            mean_tokens: 0,
            cost_usd: 0.0,
        });
    }

    results
}
```

---

## A/B Comparison Framework

```rust
// benches/sciengine/comparison.rs

/// Compare raw LLM output vs ThinkTools-enhanced output
pub struct ABComparison {
    pub query: String,
    pub baseline_response: String,
    pub baseline_score: f64,
    pub enhanced_response: String,
    pub enhanced_score: f64,
    pub profile: Profile,
    pub delta: f64,
    pub latency_ratio: f64,  // enhanced/baseline
    pub cost_ratio: f64,     // enhanced/baseline
}

/// Run A/B comparison across a benchmark
pub async fn run_ab_comparison(
    benchmark: &impl Benchmark,
    profile: Profile,
    trials: usize,
) -> ABComparisonResult {
    let mut baseline_correct = 0;
    let mut enhanced_correct = 0;
    let mut total = 0;

    for sample in benchmark.samples() {
        for _ in 0..trials {
            // Run baseline (no ThinkTools)
            let baseline = run_raw_query(&sample.prompt).await;
            if sample.evaluate(&baseline.output) {
                baseline_correct += 1;
            }

            // Run enhanced (with ThinkTools)
            let enhanced = execute_with_profile(&sample.prompt, profile).await;
            if sample.evaluate(&enhanced.output) {
                enhanced_correct += 1;
            }

            total += 1;
        }
    }

    ABComparisonResult {
        benchmark_name: benchmark.name(),
        profile,
        baseline_accuracy: baseline_correct as f64 / total as f64,
        enhanced_accuracy: enhanced_correct as f64 / total as f64,
        delta: (enhanced_correct - baseline_correct) as f64 / total as f64,
        is_significant: is_statistically_significant(
            baseline_correct, enhanced_correct, total
        ),
    }
}

/// Statistical significance test (McNemar's test)
fn is_statistically_significant(
    baseline_correct: usize,
    enhanced_correct: usize,
    total: usize,
) -> bool {
    // McNemar's test for paired data
    // H0: P(enhanced correct | baseline wrong) = P(baseline correct | enhanced wrong)
    todo!()
}
```

---

## Report Generation

### Markdown Report Template

````markdown
# ReasonKit Benchmark Report

**Date:** {date}
**Model:** {model}
**Commit:** {commit_hash}

## Summary

| Benchmark | Profile | Accuracy | 95% CI | Delta vs Baseline |
| --------- | ------- | -------- | ------ | ----------------- |

{summary_table}

## Detailed Results

### {benchmark_name}

**Best Profile:** {best_profile} ({best_accuracy}%)

| Profile | Accuracy | 95% CI | Latency (ms) | Cost ($) |
| ------- | -------- | ------ | ------------ | -------- |

{profile_table}

**Where ThinkTools Helped:**
{helped_examples}

**Where ThinkTools Hurt:**
{hurt_examples}

## Methodology

- Trials per sample: {trials}
- Temperature: {temperature}
- Confidence intervals: Wilson score (95%)
- Statistical test: McNemar's (p < 0.05)

## Reproducibility

```bash
# Reproduce this benchmark
{reproduction_command}
```
````

````

### CI/CD Integration

```yaml
# .github/workflows/benchmarks.yml

name: Benchmark Suite

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:
    inputs:
      benchmark:
        description: 'Benchmark to run'
        required: false
        default: 'all'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run Benchmarks
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cargo run --release --bin sciengine -- \
            --benchmark ${{ github.event.inputs.benchmark || 'quick' }} \
            --output results/benchmark_$(date +%Y%m%d).json

      - name: Generate Report
        run: |
          python scripts/benchmark/report_generator.py \
            results/benchmark_*.json \
            --output results/BENCHMARK_REPORT.md

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results/

      - name: Comment on PR (if PR)
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('results/BENCHMARK_REPORT.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '## Benchmark Results\n\n' + report
            });
````

---

## CLI Interface

```bash
# Full benchmark suite (all benchmarks, 10 trials)
cargo run --release --bin sciengine -- --full

# Quick smoke test (10 samples, 3 trials)
cargo run --release --bin sciengine -- --quick

# Single benchmark
cargo run --release --bin sciengine -- --benchmark logiqa

# Compare profiles
cargo run --release --bin sciengine -- \
    --benchmark game_of_24 \
    --profiles quick,balanced,deep,paranoid \
    --trials 10

# A/B comparison
cargo run --release --bin sciengine -- \
    --benchmark logiqa \
    --ab-compare \
    --profile balanced

# Generate report
cargo run --release --bin sciengine -- \
    --input results/*.json \
    --report markdown \
    --output BENCHMARK_REPORT.md
```

---

## Mandatory Disclosures

Every published benchmark result MUST include:

```yaml
methodology:
  model: "claude-sonnet-4-20250514"
  model_commit: "abc123"
  temperature: 0.0
  max_tokens: 4096
  trials_per_sample: 10

hardware:
  gpu: "None (API-based)"
  cpu: "AMD EPYC 7763"
  ram: "32GB"

software:
  rust_version: "1.83.0"
  reasonkit_version: "1.0.0"
  reasonkit_commit: "def456"

reproducibility:
  command: "cargo run --release --bin sciengine -- --full"
  seed: 42
  date: "2025-12-28"
```

---

## Interpretation Guide

### What Results Mean

| Delta     | Interpretation         | Action                         |
| --------- | ---------------------- | ------------------------------ |
| > +10%    | Strong improvement     | Highlight in marketing         |
| +5-10%    | Meaningful improvement | Document trade-offs            |
| +1-5%     | Marginal               | May not justify cost           |
| -1 to +1% | No difference          | Honest: "no measurable change" |
| < -1%     | Degradation            | Document when NOT to use       |

### Cost-Benefit Analysis

```
Value = (Accuracy_delta * Value_per_correct) - (Cost_delta)

Example:
- LogiQA: +8% accuracy with balanced profile
- Baseline cost: $0.002/query
- ThinkTools cost: $0.008/query (4x tokens)
- Value per correct answer: $0.10

Value = (0.08 * $0.10) - ($0.006) = $0.008 - $0.006 = $0.002 net positive

BUT: Only if accuracy improvement matters more than latency.
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Implement statistical utilities (Wilson CI, McNemar)
- [ ] Create benchmark trait/interface
- [ ] Implement Game of 24 (marketing demo)
- [ ] Basic CLI runner

### Phase 2: Core Benchmarks (Week 3-4)

- [ ] LogiQA 2.0 integration
- [ ] MMLU-Pro integration
- [ ] BIG-Bench Hard (subset)
- [ ] A/B comparison framework

### Phase 3: Reporting (Week 5)

- [ ] Markdown report generator
- [ ] HTML dashboard
- [ ] CI/CD integration

### Phase 4: Community Release (Week 6)

- [ ] Public benchmark results
- [ ] Reproducibility documentation
- [ ] Leaderboard (optional)

---

## Sources

- [Tree of Thoughts (Yao et al.)](https://arxiv.org/abs/2305.10601)
- [Chain-of-Thought Prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Stanford HELM](https://crfm.stanford.edu/helm/)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [MMLU-Pro (NeurIPS 2024)](https://github.com/TIGER-AI-Lab/MMLU-Pro)
- [LogiQA 2.0](https://huggingface.co/datasets/lucasmccabe/logiqa-2.0)

---

_SciEngine Benchmark Suite v1.0 | ReasonKit Core_
_"No claims without evidence."_
