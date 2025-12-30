# ReasonKit Benchmark Validation Guide

> **Purpose:** Validate benchmark performance with real data
> **Version:** 1.0.0
> **Status:** Production-Ready
> **Task:** 119 (P0-CRITICAL)

---

## Executive Summary

This guide provides a comprehensive process for validating ReasonKit's benchmark performance claims using real data. All benchmarks must be reproducible, statistically rigorous, and honestly report both improvements and regressions.

**Core Principle:** No claims without evidence. All performance metrics must be validated with real data.

---

## Validation Requirements

### 1. Statistical Rigor

- **Minimum Sample Size:** 20 problems per benchmark (100+ for publication)
- **Confidence Intervals:** Report 95% CI on all accuracy metrics
- **Significance Testing:** p < 0.05 for any improvement claims
- **Multiple Trials:** 5-10 trials per problem for Self-Consistency validation

### 2. Reproducibility

- **Fixed Seeds:** All random operations use fixed seeds
- **Documented Environment:** Model version, API endpoint, temperature, system specs
- **Public Prompts:** All benchmark prompts are publicly available
- **Version Control:** Benchmark code and results are versioned

### 3. Honest Reporting

- **Publish All Results:** Including negative ones
- **Cost Analysis:** Token usage and latency reported alongside accuracy
- **Failure Cases:** Document where ThinkTools perform worse
- **Baseline Comparison:** Always compare against raw LLM performance

---

## Validation Process

### Phase 1: Infrastructure Validation

**Objective:** Verify benchmark infrastructure works correctly.

```bash
# 1. Build release binary
cd reasonkit-core
cargo build --release

# 2. Run Criterion benchmarks (performance, no API calls)
cargo bench

# 3. Verify all benchmarks complete without errors
# Expected: All benchmarks report < 5ms for core loops

# 4. Check HTML reports
open target/criterion/report/index.html
```

**Success Criteria:**

- ✅ All Criterion benchmarks complete
- ✅ No compilation errors
- ✅ Performance targets met (< 5ms for core loops)
- ✅ HTML reports generated

---

### Phase 2: Mock Mode Validation

**Objective:** Validate benchmark logic with mock LLM responses.

```bash
# Run benchmark with mock LLM (no API calls, fast)
cargo run --release --bin bench -- \
    --benchmark gsm8k \
    --samples 5 \
    --self-consistency 3 \
    --mock \
    --profile balanced \
    --verbose

# Expected output:
# - Benchmark completes successfully
# - Results show baseline vs Self-Consistency comparison
# - Statistical metrics calculated correctly
```

**Success Criteria:**

- ✅ Benchmark completes without errors
- ✅ Results structure is valid JSON
- ✅ Accuracy calculations are correct
- ✅ Statistical significance is computed

---

### Phase 3: Real Data Validation (Small Sample)

**Objective:** Validate with real LLM calls using small sample size.

**Prerequisites:**

- API key configured (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`)
- Budget allocated for API calls (~$0.50-2.00 for 20 samples)

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run small validation (20 samples, 3 SC samples)
cargo run --release --bin bench -- \
    --benchmark gsm8k \
    --samples 20 \
    --self-consistency 3 \
    --mock false \
    --profile balanced \
    --output json \
    > validation_results.json

# Verify results
cat validation_results.json | jq '.baseline_accuracy, .sc_accuracy, .accuracy_delta'
```

**Success Criteria:**

- ✅ Real LLM calls complete successfully
- ✅ Results show meaningful accuracy metrics
- ✅ Token usage and latency are recorded
- ✅ Self-Consistency shows improvement (or honest reporting if not)

---

### Phase 4: Full Validation Suite

**Objective:** Run comprehensive validation with multiple benchmarks and profiles.

```bash
# Create validation script
cat > scripts/validate_benchmarks.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        ReasonKit Benchmark Validation Suite                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check API key
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "⚠️  WARNING: No API key found. Running in mock mode only."
    MOCK_FLAG="--mock"
else
    echo "✅ API key found. Running with real LLM calls."
    MOCK_FLAG=""
fi

# Create results directory
mkdir -p benchmark_validation_results
RESULTS_DIR="benchmark_validation_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📊 Results will be saved to: $RESULTS_DIR"
echo ""

# Run GSM8K validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Benchmark 1/2: GSM8K (Math Reasoning)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo run --release --bin bench -- \
    --benchmark gsm8k \
    --samples 20 \
    --self-consistency 5 \
    $MOCK_FLAG \
    --profile balanced \
    --output json \
    > "$RESULTS_DIR/gsm8k_balanced.json"

echo ""
echo "✅ GSM8K validation complete"
echo ""

# Run ARC-C validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Benchmark 2/2: ARC-C (Science Reasoning)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo run --release --bin bench -- \
    --benchmark arc-c \
    --samples 10 \
    --self-consistency 3 \
    $MOCK_FLAG \
    --profile balanced \
    --output json \
    > "$RESULTS_DIR/arc_c_balanced.json"

echo ""
echo "✅ ARC-C validation complete"
echo ""

# Generate summary report
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command -v jq >/dev/null; then
    echo "GSM8K Results:"
    cat "$RESULTS_DIR/gsm8k_balanced.json" | jq '{
        baseline_accuracy: .baseline_accuracy,
        sc_accuracy: .sc_accuracy,
        accuracy_delta: .accuracy_delta,
        is_significant: .is_significant,
        token_multiplier: .token_multiplier
    }'

    echo ""
    echo "ARC-C Results:"
    cat "$RESULTS_DIR/arc_c_balanced.json" | jq '{
        baseline_accuracy: .baseline_accuracy,
        sc_accuracy: .sc_accuracy,
        accuracy_delta: .accuracy_delta,
        is_significant: .is_significant,
        token_multiplier: .token_multiplier
    }'
else
    echo "Install 'jq' for formatted summary: apt install jq / brew install jq"
    echo "Raw results available in: $RESULTS_DIR/"
fi

echo ""
echo "✅ Validation complete! Results saved to: $RESULTS_DIR"
EOF

chmod +x scripts/validate_benchmarks.sh

# Run validation
./scripts/validate_benchmarks.sh
```

**Success Criteria:**

- ✅ All benchmarks complete successfully
- ✅ Results show statistical significance where applicable
- ✅ Token costs are reasonable (< 10x multiplier for SC)
- ✅ Accuracy improvements are validated (or honestly reported if negative)

---

## Validation Checklist

### Pre-Validation

- [ ] Release build compiles without errors
- [ ] All Criterion benchmarks pass (< 5ms target)
- [ ] API keys configured (for real data validation)
- [ ] Budget allocated for API calls
- [ ] Results directory created

### Mock Mode Validation

- [ ] Mock benchmarks complete successfully
- [ ] Results structure is valid
- [ ] Accuracy calculations are correct
- [ ] Statistical metrics computed

### Real Data Validation

- [ ] Small sample (20 problems) completes
- [ ] Real LLM calls succeed
- [ ] Token usage recorded
- [ ] Latency measured
- [ ] Results show improvement or honest reporting

### Full Suite Validation

- [ ] Multiple benchmarks run (GSM8K, ARC-C)
- [ ] Multiple profiles tested (quick, balanced, deep)
- [ ] Self-Consistency validated
- [ ] Results published (including negative ones)

---

## Expected Results

### Performance Targets

| Metric                       | Target   | Validation Method     |
| ---------------------------- | -------- | --------------------- |
| **Core Loop Latency**        | < 5ms    | Criterion benchmarks  |
| **Protocol Overhead**        | < 50ms   | Mock mode benchmarks  |
| **Accuracy Improvement**     | > +5%    | Real data benchmarks  |
| **Token Multiplier (SC)**    | < 10x    | Cost analysis         |
| **Statistical Significance** | p < 0.05 | Z-test on proportions |

### Sample Validation Output

```json
{
  "benchmark": "GSM8K",
  "profile": "balanced",
  "total_problems": 20,
  "baseline_accuracy": 0.75,
  "sc_accuracy": 0.85,
  "accuracy_delta": 0.1,
  "accuracy_delta_percent": 13.3,
  "is_significant": true,
  "p_value_approx": 0.023,
  "token_multiplier": 5.2,
  "sc_avg_agreement": 0.82
}
```

**Interpretation:**

- ✅ **Accuracy Delta:** +10% (meaningful improvement)
- ✅ **Statistical Significance:** p = 0.023 < 0.05 (significant)
- ⚠️ **Token Multiplier:** 5.2x (acceptable, but monitor cost)
- ✅ **Agreement:** 82% (good consensus)

---

## Troubleshooting

### Common Issues

**1. API Key Not Found**

```bash
# Check environment
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set if missing
export ANTHROPIC_API_KEY=your_key_here
```

**2. Benchmark Timeout**

```bash
# Increase timeout in benchmark code
# Or run with smaller sample size
cargo run --release --bin bench -- --samples 10
```

**3. Compilation Errors**

```bash
# Clean and rebuild
cargo clean
cargo build --release
```

**4. Invalid Results**

```bash
# Check JSON validity
cat results.json | jq .

# Verify structure
cat results.json | jq 'keys'
```

---

## Continuous Validation

### CI/CD Integration

Add to `.github/workflows/benchmark.yml`:

```yaml
- name: Validate Benchmarks (Mock Mode)
  run: |
    cargo run --release --bin bench -- \
      --benchmark gsm8k \
      --samples 5 \
      --mock \
      --output json \
      > validation_mock.json

    # Verify results structure
    cat validation_mock.json | jq '.baseline_accuracy, .sc_accuracy'
```

### Nightly Validation

Run full validation suite nightly (with real API calls):

```bash
# Cron job or scheduled CI
0 2 * * * cd /path/to/reasonkit-core && ./scripts/validate_benchmarks.sh
```

---

## Reporting Results

### Validation Report Template

```markdown
# Benchmark Validation Report

**Date:** 2025-12-29
**Version:** reasonkit-core v1.0.0
**Environment:** [System specs, model versions]

## Results Summary

| Benchmark | Baseline | SC  | Delta | Significant | Status      |
| --------- | -------- | --- | ----- | ----------- | ----------- |
| GSM8K     | 75%      | 85% | +10%  | ✅ Yes      | ✅ PASS     |
| ARC-C     | 60%      | 65% | +5%   | ⚠️ No       | ⚠️ MARGINAL |

## Cost Analysis

- **Token Usage:** 5.2x multiplier for Self-Consistency
- **Latency:** +150ms average overhead
- **Cost per 100 problems:** ~$2.50

## Conclusion

✅ Benchmarks validated with real data. Accuracy improvements confirmed for GSM8K.
⚠️ ARC-C shows marginal improvement (not statistically significant).
```

---

## Next Steps

1. **Run Initial Validation:** Execute Phase 1-3 validation
2. **Document Results:** Create validation report
3. **Publish Results:** Share results (including negative ones)
4. **Iterate:** Use results to improve ThinkTools
5. **Continuous Monitoring:** Set up nightly validation

---

## References

- **Benchmark Design:** `docs/BENCHMARK_SUITE_DESIGN.md`
- **Performance Guide:** `docs/PERFORMANCE_BENCHMARKING_GUIDE.md`
- **Benchmark Code:** `src/bin/bench.rs`
- **Criterion Benchmarks:** `benches/`

---

**Last Updated:** 2025-12-29  
**Status:** ✅ Validation Guide Complete  
**Next:** Run Phase 1-3 validation and document results
