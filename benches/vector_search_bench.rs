//! Vector Search Benchmarks at Various Scales
//!
//! Comprehensive benchmarks for vector similarity search operations
//! testing performance across different corpus sizes and query patterns.
//!
//! ## Performance Targets
//!
//! | Scale      | Corpus Size | Target Latency | Target QPS |
//! |------------|-------------|----------------|------------|
//! | Small      | 100         | < 1ms          | > 1000     |
//! | Medium     | 1,000       | < 5ms          | > 200      |
//! | Large      | 10,000      | < 20ms         | > 50       |
//! | X-Large    | 100,000     | < 100ms        | > 10       |
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench vector_search_bench --features memory
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// MOCK VECTOR DATABASE
// =============================================================================

/// Mock embedding vector (simulates 384-dim BGE-M3 output)
fn generate_embedding(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed + i) as f32 * 0.00731).sin())
        .collect()
}

/// Mock vector index for benchmarking (no external dependencies)
struct MockVectorIndex {
    vectors: Vec<(usize, Vec<f32>)>,
    dim: usize,
}

impl MockVectorIndex {
    fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
        }
    }

    fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(capacity),
            dim,
        }
    }

    fn insert(&mut self, id: usize, vector: Vec<f32>) {
        self.vectors.push((id, vector));
    }

    fn insert_batch(&mut self, vectors: Vec<(usize, Vec<f32>)>) {
        self.vectors.extend(vectors);
    }

    /// Brute-force similarity search (baseline)
    fn search_brute_force(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| (*id, cosine_similarity(query, vec)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Optimized search with early termination
    fn search_optimized(&self, query: &[f32], top_k: usize, threshold: f32) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = Vec::with_capacity(top_k * 2);

        for (id, vec) in &self.vectors {
            let score = cosine_similarity(query, vec);
            if score >= threshold {
                results.push((*id, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Partitioned search (simulates sharded index)
    fn search_partitioned(
        &self,
        query: &[f32],
        top_k: usize,
        num_partitions: usize,
    ) -> Vec<(usize, f32)> {
        let partition_size = self.vectors.len() / num_partitions;
        let mut all_results: Vec<(usize, f32)> = Vec::new();

        for p in 0..num_partitions {
            let start = p * partition_size;
            let end = if p == num_partitions - 1 {
                self.vectors.len()
            } else {
                (p + 1) * partition_size
            };

            let mut partition_results: Vec<(usize, f32)> = self.vectors[start..end]
                .iter()
                .map(|(id, vec)| (*id, cosine_similarity(query, vec)))
                .collect();

            partition_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            partition_results.truncate(top_k);
            all_results.extend(partition_results);
        }

        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);
        all_results
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

/// Compute dot product (for normalized vectors, equivalent to cosine)
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

fn create_test_index(size: usize, dim: usize) -> MockVectorIndex {
    let mut index = MockVectorIndex::with_capacity(dim, size);
    for i in 0..size {
        index.insert(i, generate_embedding(i, dim));
    }
    index
}

fn create_query_set(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_embedding(1000000 + i, dim))
        .collect()
}

// =============================================================================
// BENCHMARK: CORPUS SIZE SCALING
// =============================================================================

/// Benchmark vector search latency at different corpus sizes
fn bench_corpus_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_scale");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let dim = 384;
    let top_k = 10;
    let corpus_sizes = vec![100, 500, 1_000, 5_000, 10_000, 50_000];

    for size in corpus_sizes {
        let index = create_test_index(size, dim);
        let query = generate_embedding(999999, dim);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("brute_force", size), &size, |b, _| {
            b.iter(|| {
                let results = index.search_brute_force(black_box(&query), top_k);
                black_box(results)
            })
        });

        // Optimized search only for larger corpora
        if size >= 1000 {
            group.bench_with_input(
                BenchmarkId::new("optimized_threshold", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let results = index.search_optimized(black_box(&query), top_k, 0.5);
                        black_box(results)
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: TOP-K VARIATIONS
// =============================================================================

/// Benchmark how top_k affects search performance
fn bench_topk_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_topk");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let corpus_size = 10_000;
    let index = create_test_index(corpus_size, dim);
    let query = generate_embedding(999999, dim);

    let topk_values = vec![1, 5, 10, 20, 50, 100, 200];

    for k in topk_values {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let results = index.search_brute_force(black_box(&query), k);
                black_box(results)
            })
        });
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: DIMENSION IMPACT
// =============================================================================

/// Benchmark how embedding dimension affects search performance
fn bench_dimension_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_dimension");
    group.measurement_time(Duration::from_secs(10));

    let corpus_size = 5_000;
    let top_k = 10;
    let dimensions = vec![128, 256, 384, 512, 768, 1024, 1536];

    for dim in dimensions {
        let index = create_test_index(corpus_size, dim);
        let query = generate_embedding(999999, dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                let results = index.search_brute_force(black_box(&query), top_k);
                black_box(results)
            })
        });
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: SIMILARITY METRICS
// =============================================================================

/// Benchmark different similarity metrics
fn bench_similarity_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_metrics");

    let dim = 384;
    let vec_a: Vec<f32> = generate_embedding(1, dim);
    let vec_b: Vec<f32> = generate_embedding(2, dim);

    group.bench_function("cosine_similarity", |b| {
        b.iter(|| {
            let score = cosine_similarity(black_box(&vec_a), black_box(&vec_b));
            black_box(score)
        })
    });

    group.bench_function("dot_product", |b| {
        b.iter(|| {
            let score = dot_product(black_box(&vec_a), black_box(&vec_b));
            black_box(score)
        })
    });

    group.bench_function("euclidean_distance", |b| {
        b.iter(|| {
            let dist = euclidean_distance(black_box(&vec_a), black_box(&vec_b));
            black_box(dist)
        })
    });

    // Batch similarity computation
    let batch_size = 100;
    let batch_vectors: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| generate_embedding(i, dim))
        .collect();

    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("batch_cosine_100", |b| {
        b.iter(|| {
            let scores: Vec<f32> = batch_vectors
                .iter()
                .map(|v| cosine_similarity(&vec_a, v))
                .collect();
            black_box(scores)
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: BATCH QUERIES
// =============================================================================

/// Benchmark batch query processing
fn bench_batch_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_batch");
    group.measurement_time(Duration::from_secs(15));

    let dim = 384;
    let corpus_size = 10_000;
    let top_k = 10;
    let index = create_test_index(corpus_size, dim);

    let batch_sizes = vec![1, 5, 10, 20, 50];

    for batch_size in batch_sizes {
        let queries = create_query_set(batch_size, dim);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<Vec<(usize, f32)>> = queries
                        .iter()
                        .map(|q| index.search_brute_force(q, top_k))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel batch search using rayon
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &batch_size,
            |b, _| {
                use rayon::prelude::*;
                b.iter(|| {
                    let results: Vec<Vec<(usize, f32)>> = queries
                        .par_iter()
                        .map(|q| index.search_brute_force(q, top_k))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: PARTITIONED/SHARDED SEARCH
// =============================================================================

/// Benchmark partitioned search (simulating sharded vector DB)
fn bench_partitioned_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_partitioned");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let corpus_size = 50_000;
    let top_k = 10;
    let index = create_test_index(corpus_size, dim);
    let query = generate_embedding(999999, dim);

    let partition_counts = vec![1, 2, 4, 8, 16];

    for partitions in partition_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(partitions),
            &partitions,
            |b, &p| {
                b.iter(|| {
                    let results = index.search_partitioned(black_box(&query), top_k, p);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: INDEX BUILDING
// =============================================================================

/// Benchmark index construction time
fn bench_index_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_index_building");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    let dim = 384;
    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        // Single insert
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("single_insert", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut index = MockVectorIndex::with_capacity(dim, size);
                    for i in 0..size {
                        index.insert(i, generate_embedding(i, dim));
                    }
                    black_box(index.len())
                })
            },
        );

        // Batch insert
        group.bench_with_input(BenchmarkId::new("batch_insert", size), &size, |b, &size| {
            let vectors: Vec<(usize, Vec<f32>)> =
                (0..size).map(|i| (i, generate_embedding(i, dim))).collect();

            b.iter(|| {
                let mut index = MockVectorIndex::with_capacity(dim, size);
                index.insert_batch(vectors.clone());
                black_box(index.len())
            })
        });
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: MEMORY EFFICIENCY
// =============================================================================

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_memory");

    let dim = 384;

    // Allocation patterns
    group.bench_function("allocate_embedding", |b| {
        b.iter(|| {
            let vec: Vec<f32> = vec![0.0f32; dim];
            black_box(vec)
        })
    });

    group.bench_function("generate_embedding", |b| {
        b.iter(|| {
            let vec = generate_embedding(black_box(42), dim);
            black_box(vec)
        })
    });

    // Clone vs reference patterns
    let embedding = generate_embedding(42, dim);

    group.bench_function("clone_embedding", |b| {
        b.iter(|| {
            let cloned = embedding.clone();
            black_box(cloned)
        })
    });

    group.bench_function("slice_reference", |b| {
        b.iter(|| {
            let slice: &[f32] = black_box(&embedding);
            black_box(slice.len())
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: CONCURRENT SEARCH
// =============================================================================

/// Benchmark concurrent search operations
fn bench_concurrent_search(c: &mut Criterion) {
    use std::sync::Arc;

    let mut group = c.benchmark_group("vector_search_concurrent");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);

    let dim = 384;
    let corpus_size = 10_000;
    let top_k = 10;
    let index = Arc::new(create_test_index(corpus_size, dim));

    let concurrency_levels = vec![1, 2, 4, 8];

    for concurrency in concurrency_levels {
        let queries: Vec<Vec<f32>> = create_query_set(concurrency, dim);

        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, _| {
                b.iter(|| {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = queries
                            .iter()
                            .map(|q| {
                                let idx = index.clone();
                                s.spawn(move || idx.search_brute_force(q, top_k))
                            })
                            .collect();

                        let results: Vec<_> =
                            handles.into_iter().map(|h| h.join().unwrap()).collect();
                        black_box(results)
                    })
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: FILTER PERFORMANCE
// =============================================================================

/// Benchmark filtered search (metadata filtering)
fn bench_filtered_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_filtered");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let corpus_size = 10_000;
    let top_k = 10;

    // Create index with category metadata
    struct VectorWithMeta {
        id: usize,
        vector: Vec<f32>,
        category: usize,
    }

    let vectors: Vec<VectorWithMeta> = (0..corpus_size)
        .map(|i| VectorWithMeta {
            id: i,
            vector: generate_embedding(i, dim),
            category: i % 10, // 10 categories
        })
        .collect();

    let query = generate_embedding(999999, dim);

    // Unfiltered search
    group.bench_function("unfiltered", |b| {
        b.iter(|| {
            let mut scores: Vec<(usize, f32)> = vectors
                .iter()
                .map(|v| (v.id, cosine_similarity(&query, &v.vector)))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);
            black_box(scores)
        })
    });

    // Filtered search (single category)
    let target_category = 5;
    group.bench_function("filtered_10pct", |b| {
        b.iter(|| {
            let mut scores: Vec<(usize, f32)> = vectors
                .iter()
                .filter(|v| v.category == target_category)
                .map(|v| (v.id, cosine_similarity(&query, &v.vector)))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);
            black_box(scores)
        })
    });

    // Filtered search (multiple categories - 50%)
    let target_categories = vec![0, 1, 2, 3, 4];
    group.bench_function("filtered_50pct", |b| {
        b.iter(|| {
            let mut scores: Vec<(usize, f32)> = vectors
                .iter()
                .filter(|v| target_categories.contains(&v.category))
                .map(|v| (v.id, cosine_similarity(&query, &v.vector)))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);
            black_box(scores)
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = scale_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .with_plots();
    targets = bench_corpus_scale, bench_topk_variations
);

criterion_group!(
    name = dimension_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2));
    targets = bench_dimension_impact, bench_similarity_metrics
);

criterion_group!(
    name = batch_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3));
    targets = bench_batch_queries, bench_partitioned_search
);

criterion_group!(
    name = system_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2));
    targets = bench_index_building, bench_memory_patterns
);

criterion_group!(
    name = concurrent_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(30);
    targets = bench_concurrent_search, bench_filtered_search
);

criterion_main!(
    scale_benches,
    dimension_benches,
    batch_benches,
    system_benches,
    concurrent_benches
);
