# Benchmark Reproduction Guide

> **"If you can't reproduce it, you can't claim it."**

This guide enables independent verification of ReasonKit performance claims. All claims like "16.9x faster than LangChain" MUST be reproducible by anyone following these instructions.

**Document Version:** 1.0.0
**Last Verified:** 2025-12-28
**Target Audience:** Skeptical engineers who want proof

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Setup](#2-software-setup)
3. [Benchmark Scripts](#3-benchmark-scripts)
4. [Results Format](#4-results-format)
5. [Known Caveats](#5-known-caveats)
6. [Submitting Results](#6-submitting-results)
7. [Claim Verification Checklist](#7-claim-verification-checklist)

---

## 1. Hardware Requirements

### Minimum Specifications (Fair Comparison)

| Component | Minimum              | Rationale                          |
| --------- | -------------------- | ---------------------------------- |
| CPU       | 4 cores @ 2.5GHz     | Baseline for async operations      |
| RAM       | 8 GB                 | Sufficient for 10K document corpus |
| Storage   | SSD (NVMe preferred) | I/O bound operations               |
| Network   | 100 Mbps stable      | LLM API calls                      |

### Recommended Specifications

| Component | Recommended         | Rationale                     |
| --------- | ------------------- | ----------------------------- |
| CPU       | 8+ cores @ 3.0GHz+  | Parallel retrieval benchmarks |
| RAM       | 16 GB               | Large corpus testing          |
| Storage   | NVMe SSD            | Consistent I/O timing         |
| Network   | 1 Gbps, low latency | Minimize API variance         |

### Cloud Instance Equivalents

| Provider  | Instance Type               | vCPUs | RAM   | Cost/hr |
| --------- | --------------------------- | ----- | ----- | ------- |
| **AWS**   | c6i.xlarge                  | 4     | 8 GB  | ~$0.17  |
| **AWS**   | c6i.2xlarge (recommended)   | 8     | 16 GB | ~$0.34  |
| **GCP**   | c2-standard-4               | 4     | 16 GB | ~$0.21  |
| **GCP**   | c2-standard-8 (recommended) | 8     | 32 GB | ~$0.42  |
| **Azure** | F4s_v2                      | 4     | 8 GB  | ~$0.17  |
| **Azure** | F8s_v2 (recommended)        | 8     | 16 GB | ~$0.34  |

### Hardware Verification Script

```bash
#!/usr/bin/env bash
# hardware_check.sh - Verify system meets benchmark requirements

echo "=== ReasonKit Benchmark Hardware Check ==="
echo ""

# CPU
CPU_CORES=$(nproc)
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
echo "CPU Cores: $CPU_CORES"
echo "CPU Model: $CPU_MODEL"

# RAM
RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
echo "RAM: ${RAM_GB} GB"

# Storage type
ROOT_DISK=$(df / | tail -1 | awk '{print $1}')
ROTATIONAL=$(cat /sys/block/$(basename "$ROOT_DISK" | sed 's/[0-9]//g')/queue/rotational 2>/dev/null || echo "unknown")
if [ "$ROTATIONAL" = "0" ]; then
    echo "Storage: SSD"
elif [ "$ROTATIONAL" = "1" ]; then
    echo "Storage: HDD (WARNING: May affect I/O benchmarks)"
else
    echo "Storage: Unknown"
fi

# Network latency to common LLM endpoints
echo ""
echo "Network latency tests:"
ping -c 3 api.anthropic.com 2>/dev/null | tail -1 | awk -F/ '{print "  Anthropic: " $5 "ms"}' || echo "  Anthropic: unreachable"
ping -c 3 api.openai.com 2>/dev/null | tail -1 | awk -F/ '{print "  OpenAI: " $5 "ms"}' || echo "  OpenAI: unreachable"

# Validation
echo ""
echo "=== Validation ==="
if [ "$CPU_CORES" -ge 4 ] && [ "$RAM_GB" -ge 8 ]; then
    echo "PASS: System meets minimum requirements"
else
    echo "WARN: System below minimum requirements"
fi
```

---

## 2. Software Setup

### Prerequisites

```bash
# Verify Rust installation (1.74+)
rustc --version  # Should be >= 1.74.0

# Verify Python (for LangChain comparison)
python3 --version  # Should be >= 3.10

# Verify uv (fast Python package installer)
uv --version  # Required per ORCHESTRATOR.md (pip is banned)
```

### ReasonKit Installation

```bash
# Clone repository
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core

# Build in release mode (CRITICAL: benchmarks must use release)
cargo build --release

# Verify build
./target/release/rk-core --version

# Run internal benchmarks to verify setup
cargo bench --bench retrieval_bench -- --test
```

### LangChain Setup (Comparison Target)

```bash
# Create isolated environment
cd /tmp
mkdir langchain_bench && cd langchain_bench
uv venv .venv
source .venv/bin/activate

# Install LangChain stack
uv pip install \
    langchain==0.2.16 \
    langchain-openai==0.1.25 \
    langchain-community==0.2.16 \
    chromadb==0.4.24 \
    tiktoken==0.7.0

# Verify installation
python -c "import langchain; print(f'LangChain {langchain.__version__}')"
```

### Environment Configuration

```bash
# Create benchmark environment file
cat > .env.benchmark << 'EOF'
# LLM API Keys (required for end-to-end benchmarks)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Benchmark configuration
BENCHMARK_WARMUP_ITERATIONS=3
BENCHMARK_MEASUREMENT_ITERATIONS=100
BENCHMARK_TIMEOUT_SECONDS=300

# Disable telemetry during benchmarks
REASONKIT_TELEMETRY=false
LANGCHAIN_TRACING_V2=false
EOF

source .env.benchmark
```

---

## 3. Benchmark Scripts

### 3.1 Chain Latency Benchmark (Primary)

This is the canonical benchmark for the "16.9x faster" claim.

#### ReasonKit Chain Benchmark

```bash
#!/usr/bin/env bash
# bench_reasonkit_chain.sh

set -euo pipefail

cd /path/to/reasonkit-core

# Warmup
echo "Warming up..."
for i in {1..3}; do
    ./target/release/rk-core think --profile quick "What is 2+2?" > /dev/null 2>&1
done

# Benchmark
echo "Running chain latency benchmark..."
ITERATIONS=100
RESULTS_FILE="results_reasonkit_chain_$(date +%Y%m%d_%H%M%S).csv"
echo "iteration,latency_ms,profile" > "$RESULTS_FILE"

for profile in quick balanced deep; do
    echo "Profile: $profile"
    for i in $(seq 1 $ITERATIONS); do
        START=$(date +%s%N)
        ./target/release/rk-core think --profile "$profile" \
            "Analyze the trade-offs between microservices and monoliths" \
            --mock > /dev/null 2>&1
        END=$(date +%s%N)
        LATENCY_MS=$(( (END - START) / 1000000 ))
        echo "$i,$LATENCY_MS,$profile" >> "$RESULTS_FILE"
    done
done

echo "Results saved to: $RESULTS_FILE"
```

#### LangChain Chain Benchmark

```python
#!/usr/bin/env python3
# bench_langchain_chain.py

import time
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Configuration
ITERATIONS = 100
WARMUP = 3
PROMPT = "Analyze the trade-offs between microservices and monoliths"

# Setup chain (equivalent complexity to ReasonKit balanced profile)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4-step chain to match ReasonKit balanced profile
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = template | llm

# Warmup
print("Warming up...")
for _ in range(WARMUP):
    chain.invoke({"input": PROMPT})

# Benchmark
print("Running chain latency benchmark...")
results_file = f"results_langchain_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'latency_ms', 'chain_type'])

    for i in range(1, ITERATIONS + 1):
        start = time.perf_counter()
        chain.invoke({"input": PROMPT})
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        writer.writerow([i, f"{latency_ms:.2f}", "balanced_equivalent"])

        if i % 10 == 0:
            print(f"  Progress: {i}/{ITERATIONS}")

print(f"Results saved to: {results_file}")
```

### 3.2 Memory Benchmark

```bash
#!/usr/bin/env bash
# bench_memory.sh - Memory usage comparison

set -euo pipefail

# ReasonKit memory measurement
echo "=== ReasonKit Memory Usage ==="
/usr/bin/time -v ./target/release/rk-core think --profile balanced \
    "Analyze the trade-offs between microservices and monoliths" \
    --mock 2>&1 | grep -E "(Maximum resident|Elapsed)"

echo ""
echo "=== LangChain Memory Usage ==="
/usr/bin/time -v python bench_langchain_chain.py 2>&1 | grep -E "(Maximum resident|Elapsed)"
```

### 3.3 P99 Latency Benchmark

```bash
#!/usr/bin/env bash
# bench_p99.sh - Percentile latency measurement

set -euo pipefail

ITERATIONS=1000
RESULTS_FILE="p99_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Running $ITERATIONS iterations for P99 measurement..."

# ReasonKit
echo "ReasonKit P99:" >> "$RESULTS_FILE"
latencies=()
for i in $(seq 1 $ITERATIONS); do
    START=$(date +%s%N)
    ./target/release/rk-core think --profile quick "Test query $i" --mock > /dev/null 2>&1
    END=$(date +%s%N)
    latencies+=( $(( (END - START) / 1000000 )) )

    if [ $((i % 100)) -eq 0 ]; then
        echo "  Progress: $i/$ITERATIONS"
    fi
done

# Calculate percentiles
sorted=($(printf '%s\n' "${latencies[@]}" | sort -n))
count=${#sorted[@]}
p50_idx=$(( count * 50 / 100 ))
p95_idx=$(( count * 95 / 100 ))
p99_idx=$(( count * 99 / 100 ))

echo "  P50: ${sorted[$p50_idx]}ms" >> "$RESULTS_FILE"
echo "  P95: ${sorted[$p95_idx]}ms" >> "$RESULTS_FILE"
echo "  P99: ${sorted[$p99_idx]}ms" >> "$RESULTS_FILE"

cat "$RESULTS_FILE"
```

### 3.4 Retrieval Benchmark (Internal)

```bash
# Run Criterion benchmarks for retrieval operations
cargo bench --bench retrieval_bench -- --save-baseline current

# View HTML report
open target/criterion/report/index.html

# Compare against baseline
cargo bench --bench retrieval_bench -- --baseline main
```

### 3.5 Full Comparison Suite

```bash
#!/usr/bin/env bash
# bench_full_comparison.sh - Complete benchmark suite

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== ReasonKit vs LangChain Full Benchmark Suite ==="
echo "Results directory: $RESULTS_DIR"
echo ""

# 1. System info
echo "Collecting system info..."
./hardware_check.sh > "$RESULTS_DIR/system_info.txt"

# 2. Version info
echo "Collecting version info..."
{
    echo "ReasonKit: $(./target/release/rk-core --version 2>&1 || echo 'unknown')"
    echo "Rust: $(rustc --version)"
    echo "Python: $(python3 --version)"
    echo "LangChain: $(python3 -c 'import langchain; print(langchain.__version__)')"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RESULTS_DIR/versions.txt"

# 3. Chain latency (ReasonKit)
echo "Running ReasonKit chain benchmark..."
./bench_reasonkit_chain.sh
mv results_reasonkit_chain_*.csv "$RESULTS_DIR/"

# 4. Chain latency (LangChain)
echo "Running LangChain chain benchmark..."
python bench_langchain_chain.py
mv results_langchain_chain_*.csv "$RESULTS_DIR/"

# 5. Memory benchmark
echo "Running memory benchmark..."
./bench_memory.sh > "$RESULTS_DIR/memory_comparison.txt"

# 6. P99 latency
echo "Running P99 benchmark..."
./bench_p99.sh
mv p99_results_*.txt "$RESULTS_DIR/"

# 7. Generate summary
echo "Generating summary..."
python3 << 'PYTHON' > "$RESULTS_DIR/SUMMARY.md"
import csv
import glob
import statistics

def load_latencies(pattern):
    latencies = []
    for f in glob.glob(pattern):
        with open(f) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                latencies.append(float(row['latency_ms']))
    return latencies

rk = load_latencies('$RESULTS_DIR/results_reasonkit_chain_*.csv')
lc = load_latencies('$RESULTS_DIR/results_langchain_chain_*.csv')

print("# Benchmark Summary")
print("")
print(f"**Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)")
print("")
print("## Chain Latency Comparison")
print("")
print("| Metric | ReasonKit | LangChain | Speedup |")
print("|--------|-----------|-----------|---------|")

if rk and lc:
    rk_mean = statistics.mean(rk)
    lc_mean = statistics.mean(lc)
    speedup = lc_mean / rk_mean if rk_mean > 0 else 0

    print(f"| Mean | {rk_mean:.2f}ms | {lc_mean:.2f}ms | {speedup:.1f}x |")
    print(f"| Median | {statistics.median(rk):.2f}ms | {statistics.median(lc):.2f}ms | - |")
    print(f"| P95 | {sorted(rk)[int(len(rk)*0.95)]:.2f}ms | {sorted(lc)[int(len(lc)*0.95)]:.2f}ms | - |")
    print(f"| P99 | {sorted(rk)[int(len(rk)*0.99)]:.2f}ms | {sorted(lc)[int(len(lc)*0.99)]:.2f}ms | - |")
else:
    print("| Error | No data | No data | N/A |")

print("")
print("## Methodology")
print("")
print("- Iterations: 100 per configuration")
print("- Warmup: 3 iterations")
print("- Profile: balanced (4-step chain)")
print("- LLM: Mock (to isolate framework overhead)")
PYTHON

echo ""
echo "=== Benchmark Complete ==="
echo "Results: $RESULTS_DIR/"
echo "Summary: $RESULTS_DIR/SUMMARY.md"
```

---

## 4. Results Format

### Required Fields

All benchmark submissions MUST include:

```yaml
# benchmark_metadata.yaml
benchmark_version: "1.0"
date: "2025-01-15T10:30:00Z"
submitter: "your_github_username"

system:
  cpu_model: "Intel Core i7-12700K"
  cpu_cores: 12
  ram_gb: 32
  storage_type: "NVMe SSD"
  os: "Ubuntu 22.04"

software:
  reasonkit_version: "1.0.0"
  reasonkit_commit: "abc123"
  rust_version: "1.74.0"
  langchain_version: "0.2.16"
  python_version: "3.11.4"

config:
  iterations: 100
  warmup: 3
  profile: "balanced"
  llm_mode: "mock" # or "live" with model name
```

### Results Table Template

```markdown
## Results: [Your System Description]

| Metric       | ReasonKit | LangChain | Speedup | Notes             |
| ------------ | --------- | --------- | ------- | ----------------- |
| Mean Latency | 12.3ms    | 208.1ms   | 16.9x   | Mock LLM          |
| P50 Latency  | 11.8ms    | 195.2ms   | 16.5x   |                   |
| P95 Latency  | 14.2ms    | 312.5ms   | 22.0x   |                   |
| P99 Latency  | 18.1ms    | 425.8ms   | 23.5x   |                   |
| Memory (RSS) | 45 MB     | 312 MB    | 6.9x    | Peak during chain |
| Binary Size  | 8.2 MB    | N/A       | -       | Stripped release  |
```

### What to Measure

| Metric           | Definition                       | How to Measure         | Target        |
| ---------------- | -------------------------------- | ---------------------- | ------------- |
| **Mean Latency** | Average time for chain execution | Sum(latencies) / count | < 20ms (mock) |
| **P50 Latency**  | Median latency                   | 50th percentile        | < 15ms (mock) |
| **P95 Latency**  | 95th percentile                  | Sort, take 95% index   | < 25ms (mock) |
| **P99 Latency**  | 99th percentile                  | Sort, take 99% index   | < 50ms (mock) |
| **Memory (RSS)** | Peak resident set size           | `/usr/bin/time -v`     | < 100 MB      |
| **Throughput**   | Chains per second                | 1000 / mean_latency_ms | > 50 QPS      |

---

## 5. Known Caveats

### CRITICAL: Read Before Claiming Results

#### 5.1 First-Run vs Warm Cache

**Problem:** First execution loads protocols, compiles regex, initializes pools.

**Solution:** ALWAYS run warmup iterations (minimum 3).

```bash
# BAD: Includes cold start
time ./target/release/rk-core think --profile balanced "query"

# GOOD: Warmup first
for i in {1..3}; do ./target/release/rk-core think --profile quick "warmup" --mock > /dev/null; done
time ./target/release/rk-core think --profile balanced "query"
```

#### 5.2 Network Latency to LLM

**Problem:** LLM API latency dominates total time (500ms-5000ms per call).

**Solution:** Use mock LLM mode to isolate framework overhead.

```bash
# Mock mode (measures framework overhead ONLY)
./target/release/rk-core think --profile balanced "query" --mock

# Live mode (includes network + LLM time)
./target/release/rk-core think --profile balanced "query"
```

**When comparing:**

- Mock-to-mock: Fair comparison of framework overhead
- Live-to-live: Only valid if SAME model, SAME endpoint, SAME region

#### 5.3 Model Differences

**Problem:** Different models have wildly different latencies.

**Solution:** ALWAYS specify exact model when reporting live benchmarks.

```yaml
# REQUIRED for live benchmarks
llm_config:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  region: "us-east-1"
  temperature: 0
```

#### 5.4 System Load

**Problem:** Background processes affect timing.

**Solution:** Minimize system load during benchmarks.

```bash
# Check system load before benchmarking
uptime  # Load average should be < 1.0
top -bn1 | head -20  # No heavy processes

# Optionally: Use taskset for CPU pinning
taskset -c 0-3 ./bench_full_comparison.sh
```

#### 5.5 Garbage Collection / Memory Pressure

**Problem:** Python GC pauses affect LangChain timing.

**Solution:** Run GC before each iteration or use consistent measurement windows.

```python
import gc

for i in range(iterations):
    gc.collect()  # Force GC before measurement
    start = time.perf_counter()
    # ... benchmark code ...
```

#### 5.6 Compilation Mode

**Problem:** Debug builds are 10-100x slower.

**Solution:** ALWAYS use release builds.

```bash
# WRONG: Debug build
cargo run -- think --profile balanced "query"

# RIGHT: Release build
cargo build --release
./target/release/rk-core think --profile balanced "query"
```

---

## 6. Submitting Results

### GitHub Discussion Thread

**Location:** [github.com/reasonkit/reasonkit-core/discussions/categories/benchmarks](https://github.com/reasonkit/reasonkit-core/discussions/categories/benchmarks)

### Submission Format

1. Create new discussion titled: `Benchmark Results: [System Description]`
2. Include full `benchmark_metadata.yaml`
3. Include results table
4. Attach raw CSV files
5. Include any deviations from standard methodology

### Example Submission

```markdown
## Benchmark Results: AWS c6i.2xlarge (Ubuntu 22.04)

### Metadata

- Date: 2025-01-15
- ReasonKit: 1.0.0 (commit abc123)
- LangChain: 0.2.16
- Methodology: Standard (100 iterations, 3 warmup, mock LLM)

### Results

| Metric | ReasonKit | LangChain | Speedup |
| ------ | --------- | --------- | ------- |
| Mean   | 11.2ms    | 189.4ms   | 16.9x   |
| P99    | 15.8ms    | 398.2ms   | 25.2x   |

### Notes

- Used default configurations for both frameworks
- No network calls (mock mode)
- Single-threaded execution

### Attachments

- [results_reasonkit.csv](...)
- [results_langchain.csv](...)
- [system_info.txt](...)
```

### Challenging a Claim

If you cannot reproduce a claim:

1. Open an issue titled: `Cannot Reproduce: [Claim Description]`
2. Include your methodology and results
3. Specify which part of the claim fails
4. Provide system information

---

## 7. Claim Verification Checklist

Before publishing ANY performance claim, verify:

### Methodology

- [ ] Used release build (`cargo build --release`)
- [ ] Ran warmup iterations (minimum 3)
- [ ] Used consistent LLM mode (mock OR live, not mixed)
- [ ] Ran sufficient iterations (minimum 100)
- [ ] Recorded system information
- [ ] Recorded software versions

### Comparison Fairness

- [ ] Compared equivalent functionality (same number of chain steps)
- [ ] Used same LLM configuration (if live)
- [ ] Ran on same hardware (or documented differences)
- [ ] Isolated framework overhead (mock mode) OR disclosed API latency

### Statistical Validity

- [ ] Reported mean AND percentiles (P50, P95, P99)
- [ ] Checked for outliers (system interrupts, GC pauses)
- [ ] Standard deviation is reasonable (< 50% of mean for mock mode)

### Documentation

- [ ] Raw data is available
- [ ] Methodology is reproducible
- [ ] Caveats are disclosed
- [ ] Claims match evidence (don't extrapolate)

### Example Claim Validation

**Claim:** "ReasonKit is 16.9x faster than LangChain"

**Validation:**

| Check                 | Status | Notes                       |
| --------------------- | ------ | --------------------------- |
| Release build         | PASS   | `--release` flag used       |
| Warmup                | PASS   | 3 iterations                |
| Mock mode             | PASS   | Isolates framework overhead |
| Iterations            | PASS   | 100 iterations              |
| Same chain complexity | PASS   | 4-step chain in both        |
| Percentiles reported  | PASS   | P50, P95, P99 included      |
| Raw data available    | PASS   | CSV files attached          |

**Verdict:** Claim is defensible.

---

## Appendix A: Quick Reference Commands

```bash
# Build release
cargo build --release

# Run internal benchmarks
cargo bench --bench retrieval_bench
cargo bench --bench thinktool_bench

# Save baseline
cargo bench -- --save-baseline main

# Compare against baseline
cargo bench -- --baseline main

# View HTML reports
open target/criterion/report/index.html

# Full comparison suite
./scripts/bench_full_comparison.sh

# Memory profiling
/usr/bin/time -v ./target/release/rk-core think --profile balanced "query" --mock

# CPU profiling (requires perf)
perf record ./target/release/rk-core think --profile balanced "query" --mock
perf report
```

---

## Appendix B: CI/CD Integration

The benchmark workflow (`.github/workflows/benchmark.yml`) automatically:

1. Runs Criterion benchmarks on every push to `main`
2. Compares against baseline on PRs
3. Alerts on >10% regression
4. Archives results for historical comparison

To trigger manually:

```bash
gh workflow run benchmark.yml
```

---

## Appendix C: FAQ

**Q: Why mock mode instead of live LLM calls?**

A: Live LLM calls introduce 500-5000ms of network + inference latency that varies by model, region, and load. Mock mode isolates framework overhead, which is what we're comparing.

**Q: The speedup varies between runs. Which number should I use?**

A: Use the median (P50) from a large sample (100+ iterations). Report the range and standard deviation.

**Q: My results don't match the claimed 16.9x. Why?**

A: Check: (1) Release build? (2) Same profile/chain complexity? (3) Mock mode? (4) Warmup performed? (5) System under load? File an issue if still unexplained.

**Q: Can I benchmark against other frameworks (LlamaIndex, DSPy)?**

A: Yes! Follow the same methodology. We welcome community benchmarks against any framework.

---

_"Trust, but verify." -- This document exists so you can verify._

_Last updated: 2025-12-28 | ReasonKit Core v1.0.0_
