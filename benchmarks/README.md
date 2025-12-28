# ReasonKit Benchmarks

**No claims without evidence.**

This directory contains reproducible benchmarks that measure the ACTUAL impact of ThinkTools on reasoning quality.

## Available Benchmarks

| Dataset | Domain            | What We Measure                      |
| ------- | ----------------- | ------------------------------------ |
| GSM8K   | Math reasoning    | Arithmetic word problems             |
| ARC-C   | Science reasoning | Grade-school science (Challenge set) |
| LogiQA  | Logical reasoning | Logical deduction problems           |

## Running Benchmarks

```bash
# Run GSM8K evaluation (default: 100 samples)
cargo run --release --bin gsm8k_eval

# Run with specific sample count
cargo run --release --bin gsm8k_eval -- --samples 500

# Run with specific profile
cargo run --release --bin gsm8k_eval -- --profile balanced

# Run all benchmarks
./scripts/run_all_benchmarks.sh
```

## What Gets Measured

For each problem, we run:

1. **Raw prompt** - Direct question to LLM
2. **ThinkTool enhanced** - Same question processed through ThinkTools

We compare:

- **Accuracy**: % of correct answers
- **Delta**: Improvement (or degradation) from ThinkTools
- **Cost**: Token usage difference
- **Latency**: Time difference

## Interpreting Results

| Delta      | Interpretation                    |
| ---------- | --------------------------------- |
| > +5%      | ✅ Meaningful improvement         |
| +1% to +5% | ⚠️ Marginal, may not justify cost |
| 0%         | ⚪ No difference                  |
| < 0%       | ❌ ThinkTools performed worse     |

## Latest Results

See `results/` directory for benchmark outputs.

**We publish ALL results, including negative ones.**

## Reproducing Results

All benchmarks are deterministic with fixed seeds. To reproduce:

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key

# Run benchmark
cargo run --release --bin gsm8k_eval -- --seed 42 --samples 100

# Compare to published results
diff results/gsm8k_latest.json your_results.json
```

## Adding New Benchmarks

1. Create `{dataset}_eval.rs` in this directory
2. Implement the evaluation loop
3. Use the common `BenchmarkResults` struct
4. Add to CI pipeline

## Philosophy

> "In God we trust. All others must bring data."
> — W. Edwards Deming

ReasonKit either improves reasoning or it doesn't. These benchmarks tell the truth.
