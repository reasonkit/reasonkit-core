# Baseline Measurements

**Date:** 2025-12-29  
**Purpose:** Establish baseline performance and accuracy metrics before implementing improvements  
**Task:** rk-project.core.benchmarks #50

---

## Overview

This document captures baseline measurements for reasonkit-core before any performance or accuracy improvements. These measurements serve as the reference point for evaluating future changes.

**Measurement Categories:**

1. **Performance Benchmarks** (Criterion.rs) - Execution time metrics
2. **Reasoning Quality Benchmarks** - Accuracy on reasoning tasks (GSM8K, ARC-C, LogiQA)

---

## Performance Benchmarks

**Target:** All core loops < 5ms

### Benchmark Suite

| Benchmark                   | Component                     | Target                | Status     |
| --------------------------- | ----------------------------- | --------------------- | ---------- |
| `retrieval_bench`           | BM25 + Hybrid search          | < 5ms for 100 docs    | ⏳ Pending |
| `fusion_bench`              | Result fusion (RRF, weighted) | < 5ms for 100 results | ⏳ Pending |
| `embedding_bench`           | Embedding operations          | < 1ms per embedding   | ⏳ Pending |
| `raptor_bench`              | RAPTOR tree operations        | < 5ms for 1000 nodes  | ⏳ Pending |
| `ingestion_bench`           | Document chunking             | < 5ms for 10KB doc    | ⏳ Pending |
| `thinktool_bench`           | ThinkTool execution           | TBD                   | ⏳ Pending |
| `rerank_bench`              | Reranking operations          | TBD                   | ⏳ Pending |
| `qdrant_optimization_bench` | Qdrant operations             | TBD                   | ⏳ Pending |

### Baseline Results

**To be populated after running:**

```bash
./scripts/run_benchmarks.sh --baseline
```

**Baseline Name:** `master` (saved in `target/criterion/`)

---

## Reasoning Quality Benchmarks

**Purpose:** Measure actual impact of ThinkTools on reasoning accuracy

### Available Datasets

| Dataset | Domain            | Samples | Status             |
| ------- | ----------------- | ------- | ------------------ |
| GSM8K   | Math reasoning    | 100+    | ⏳ Pending         |
| ARC-C   | Science reasoning | TBD     | ⏳ Not implemented |
| LogiQA  | Logical reasoning | TBD     | ⏳ Not implemented |

### Baseline Results

**GSM8K Baseline (Raw LLM):**

- Accuracy: TBD
- Mean Latency: TBD ms
- Mean Tokens: TBD
- Cost: TBD USD

**GSM8K Baseline (ThinkTools - balanced profile):**

- Accuracy: TBD
- Mean Latency: TBD ms
- Mean Tokens: TBD
- Cost: TBD USD
- Delta: TBD

**To run:**

```bash
# Raw baseline
cargo run --release --bin gsm8k_eval -- --samples 100 --profile raw

# ThinkTools baseline
cargo run --release --bin gsm8k_eval -- --samples 100 --profile balanced
```

---

## Environment

**System:**

- OS: Linux
- Rust: `rustc 1.94.0-nightly (f52090008 2025-12-10)`
- Cargo: `cargo 1.94.0-nightly (2c283a9a5 2025-12-04)`

**Build Configuration:**

- Profile: `release`
- Optimization: `-O3` (default for release)
- Target: `x86_64-unknown-linux-gnu` (or actual target)

**LLM Configuration:**

- Model: TBD (Claude Sonnet 4.0 default?)
- Temperature: 0.0 (deterministic)
- API: Anthropic (or configured provider)

---

## Comparison Methodology

### Performance Benchmarks

After improvements, compare using:

```bash
./scripts/run_benchmarks.sh --compare
```

This compares against the `master` baseline saved in Criterion.

### Reasoning Quality Benchmarks

Compare accuracy deltas:

- **> +5%**: ✅ Meaningful improvement
- **+1% to +5%**: ⚠️ Marginal, verify cost-benefit
- **0%**: ⚪ No difference
- **< 0%**: ❌ Degradation (investigate)

---

## Next Steps

1. ✅ Document baseline methodology (this file)
2. ⏳ Run performance benchmarks and save baseline
3. ⏳ Run reasoning quality benchmarks (GSM8K)
4. ⏳ Document results in this file
5. ⏳ Commit baseline to version control
6. ⏳ Reference baseline in future improvement PRs

---

## Notes

- Baseline measurements should be reproducible
- Use fixed seeds for deterministic results
- Document any environmental factors that might affect results
- Save raw data alongside summaries for future analysis

---

**Last Updated:** 2025-12-29  
**Status:** In Progress
