# ReasonKit Core - Benchmarks

This directory contains the Criterion.rs benchmark suite for reasonkit-core.

## Performance Target

**All core loops < 5ms**

## Quick Start

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench fusion_bench

# Use the helper script
cd .. && ./scripts/run_benchmarks.sh
```

## Benchmark Files

### retrieval_bench.rs

Search and retrieval performance testing.

**What it tests:**

- BM25 sparse search
- Hybrid search (sparse + dense)
- Corpus scaling (10-1000 docs)
- Concurrent queries
- Batch indexing

**Target:** < 5ms for 100 document searches

### fusion_bench.rs

Result fusion algorithm performance.

**What it tests:**

- RRF (Reciprocal Rank Fusion)
- Weighted sum fusion
- RBF (Rank-Biased Fusion)
- Multi-method fusion (2-5 methods)
- Different overlap scenarios

**Target:** < 5ms for 100 results, 2 methods

### embedding_bench.rs

Embedding operations (uses mocks to avoid API calls).

**What it tests:**

- Dense embedding generation
- Sparse embedding generation
- Batch processing
- Cache hit/miss performance
- Cosine similarity
- Parallel processing

**Target:** < 1ms per embedding, < 5ms for batch of 100

### raptor_bench.rs

RAPTOR hierarchical tree operations.

**What it tests:**

- Tree creation from leaves
- DFS and BFS traversal
- Node lookup
- Parent-child navigation
- Similarity search in tree

**Target:** < 5ms for 1000 node traversal

### ingestion_bench.rs

Document ingestion pipeline operations.

**What it tests:**

- Fixed-size chunking
- Sentence-based chunking
- Text cleaning
- Metadata extraction
- Parallel processing
- Hash computation

**Target:** < 5ms for 10KB document chunking

## Understanding Results

### Criterion Output Format

```
fusion_bench/rrf_fusion_scaling/100
                        time:   [1.1823 ms 1.2014 ms 1.2245 ms]
                        change: [-2.1234% +0.5432% +3.2109%] (p = 0.42 > 0.05)
                        No change in performance detected.
```

- **time**: [lower bound, estimate, upper bound] with 95% confidence
- **change**: Performance difference from previous run
- **p-value**: p < 0.05 indicates statistically significant change

### HTML Reports

After running benchmarks, open:

```
file://../target/criterion/report/index.html
```

## Best Practices

### Before Implementing Feature

```bash
# Save baseline
cargo bench -- --save-baseline pre-feature
```

### After Implementation

```bash
# Compare against baseline
cargo bench -- --baseline pre-feature
```

### Check for Regressions

```bash
# Look for changes > 5%
cargo bench
# Review HTML report for trends
```

## Design Principles

### 1. No External Dependencies

All benchmarks use mock implementations to avoid:

- Network calls (no Qdrant server needed)
- API calls (no OpenAI API needed)
- File I/O (all in-memory)

### 2. Deterministic

All operations produce consistent results:

- Mock embeddings based on text hash
- Seeded random number generators
- Reproducible test data

### 3. Comprehensive

Tests cover:

- Different input sizes
- Different parameter values
- Edge cases
- Concurrent scenarios
- Scaling behavior

### 4. Fast Execution

Each benchmark completes quickly:

- Mock operations are lightweight
- No real model inference
- Efficient test data generation

## Adding New Benchmarks

### Template

```rust
//! Brief description
//!
//! Performance target: < Xms

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_operation(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // Your operation here
            let result = my_function(black_box(input));
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_my_operation);
criterion_main!(benches);
```

### Steps

1. Create new file: `benches/my_bench.rs`
2. Add to `Cargo.toml`:
   ```toml
   [[bench]]
   name = "my_bench"
   harness = false
   ```
3. Implement benchmark using template above
4. Run: `cargo bench --bench my_bench`
5. Update documentation

## Troubleshooting

### Compilation Errors

```bash
# Check for type errors
cargo check --benches

# Fix and rebuild
cargo clean
cargo bench --no-run
```

### Inconsistent Results

```bash
# Ensure no background processes
# Close other applications
# Run with --save-baseline multiple times

cargo bench -- --save-baseline stable-run-1
cargo bench -- --save-baseline stable-run-2
cargo bench -- --save-baseline stable-run-3

# Compare runs - they should be very similar
```

### Benchmark Timeout

```bash
# Increase Criterion timeout in benchmark code:
group.measurement_time(Duration::from_secs(30));
```

## CI Integration

See `../.github/workflows/` for GitHub Actions integration (if configured).

Example workflow:

```yaml
- name: Run Benchmarks
  run: cargo bench --no-fail-fast
```

## References

- Main docs: `../BENCHMARKS.md`
- Summary: `../BENCHMARK_SUMMARY.md`
- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/)

---

**Total**: 5 benchmark files, ~1,600 lines of test code
**Coverage**: All critical performance paths
**Target**: < 5ms for all core loops
