//! Hybrid Search Benchmarks
//!
//! Comprehensive benchmarks for hybrid (dense + sparse) search operations
//! combining vector similarity with BM25/keyword search.
//!
//! ## Performance Targets
//!
//! | Operation          | Target Latency | Notes                              |
//! |--------------------|----------------|------------------------------------|
//! | Dense search       | < 5ms          | 10k corpus, top-10                 |
//! | Sparse search      | < 3ms          | BM25 with inverted index           |
//! | Hybrid fusion      | < 2ms          | RRF on 100 results per method      |
//! | End-to-end hybrid  | < 10ms         | Complete pipeline                  |
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench hybrid_search_bench --features memory
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// MOCK SEARCH COMPONENTS
// =============================================================================

/// Document with both dense and sparse representations
struct HybridDocument {
    id: usize,
    #[allow(dead_code)]
    text: String,
    dense_embedding: Vec<f32>,
    sparse_embedding: HashMap<String, f32>, // term -> tf-idf weight
}

/// Mock hybrid retriever
struct MockHybridRetriever {
    documents: Vec<HybridDocument>,
    inverted_index: HashMap<String, Vec<(usize, f32)>>, // term -> [(doc_id, score)]
    #[allow(dead_code)]
    dim: usize,
}

impl MockHybridRetriever {
    fn new(dim: usize) -> Self {
        Self {
            documents: Vec::new(),
            inverted_index: HashMap::new(),
            dim,
        }
    }

    fn add_document(&mut self, doc: HybridDocument) {
        let doc_id = doc.id;

        // Update inverted index
        for (term, weight) in &doc.sparse_embedding {
            self.inverted_index
                .entry(term.clone())
                .or_default()
                .push((doc_id, *weight));
        }

        self.documents.push(doc);
    }

    /// Dense vector search
    fn search_dense(&self, query_embedding: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = self
            .documents
            .iter()
            .map(|doc| {
                (
                    doc.id,
                    cosine_similarity(query_embedding, &doc.dense_embedding),
                )
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Sparse BM25-style search
    fn search_sparse(&self, query_terms: &[String], top_k: usize) -> Vec<(usize, f32)> {
        let mut doc_scores: HashMap<usize, f32> = HashMap::new();

        for term in query_terms {
            if let Some(postings) = self.inverted_index.get(term) {
                let idf = (self.documents.len() as f32 / postings.len() as f32).ln() + 1.0;
                for (doc_id, tf) in postings {
                    *doc_scores.entry(*doc_id).or_insert(0.0) += tf * idf;
                }
            }
        }

        let mut scores: Vec<(usize, f32)> = doc_scores.into_iter().collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Hybrid search with RRF fusion
    fn search_hybrid(
        &self,
        query_embedding: &[f32],
        query_terms: &[String],
        top_k: usize,
        alpha: f32,
        rrf_k: usize,
    ) -> Vec<(usize, f32)> {
        let dense_results = self.search_dense(query_embedding, top_k * 2);
        let sparse_results = self.search_sparse(query_terms, top_k * 2);

        // RRF fusion
        self.fuse_rrf(&dense_results, &sparse_results, top_k, rrf_k, alpha)
    }

    /// Reciprocal Rank Fusion
    fn fuse_rrf(
        &self,
        dense: &[(usize, f32)],
        sparse: &[(usize, f32)],
        top_k: usize,
        k: usize,
        alpha: f32,
    ) -> Vec<(usize, f32)> {
        let mut fused_scores: HashMap<usize, f32> = HashMap::new();

        // Dense contribution
        for (rank, (doc_id, _)) in dense.iter().enumerate() {
            let rrf_score = alpha * (1.0 / (k + rank + 1) as f32);
            *fused_scores.entry(*doc_id).or_insert(0.0) += rrf_score;
        }

        // Sparse contribution
        for (rank, (doc_id, _)) in sparse.iter().enumerate() {
            let rrf_score = (1.0 - alpha) * (1.0 / (k + rank + 1) as f32);
            *fused_scores.entry(*doc_id).or_insert(0.0) += rrf_score;
        }

        let mut results: Vec<(usize, f32)> = fused_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Weighted sum fusion
    fn fuse_weighted(
        &self,
        dense: &[(usize, f32)],
        sparse: &[(usize, f32)],
        top_k: usize,
        alpha: f32,
    ) -> Vec<(usize, f32)> {
        let mut fused_scores: HashMap<usize, f32> = HashMap::new();

        // Normalize dense scores
        let dense_max = dense.first().map(|(_, s)| *s).unwrap_or(1.0);
        for (doc_id, score) in dense {
            let norm_score = score / dense_max;
            *fused_scores.entry(*doc_id).or_insert(0.0) += alpha * norm_score;
        }

        // Normalize sparse scores
        let sparse_max = sparse.first().map(|(_, s)| *s).unwrap_or(1.0);
        for (doc_id, score) in sparse {
            let norm_score = score / sparse_max;
            *fused_scores.entry(*doc_id).or_insert(0.0) += (1.0 - alpha) * norm_score;
        }

        let mut results: Vec<(usize, f32)> = fused_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.documents.len()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

fn generate_embedding(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed + i) as f32 * 0.00731).sin())
        .collect()
}

fn generate_sparse_embedding(seed: usize, vocab_size: usize) -> HashMap<String, f32> {
    let num_terms = 10 + (seed % 20);
    let mut sparse = HashMap::new();

    for i in 0..num_terms {
        let term_id = (seed * 7 + i * 13) % vocab_size;
        let weight = 1.0 / (1.0 + i as f32);
        sparse.insert(format!("term_{}", term_id), weight);
    }

    sparse
}

fn create_test_retriever(size: usize, dim: usize, vocab_size: usize) -> MockHybridRetriever {
    let mut retriever = MockHybridRetriever::new(dim);

    for i in 0..size {
        let doc = HybridDocument {
            id: i,
            text: format!("Document {} about machine learning and AI", i),
            dense_embedding: generate_embedding(i, dim),
            sparse_embedding: generate_sparse_embedding(i, vocab_size),
        };
        retriever.add_document(doc);
    }

    retriever
}

#[allow(dead_code)]
fn extract_query_terms(query: &str) -> Vec<String> {
    query.split_whitespace().map(|s| s.to_lowercase()).collect()
}

// =============================================================================
// BENCHMARK: DENSE VS SPARSE SEARCH
// =============================================================================

fn bench_search_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_search_methods");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;
    let top_k = 10;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);
    let query_embedding = generate_embedding(999999, dim);
    let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

    group.bench_function("dense_only", |b| {
        b.iter(|| {
            let results = retriever.search_dense(black_box(&query_embedding), top_k);
            black_box(results)
        })
    });

    group.bench_function("sparse_only", |b| {
        b.iter(|| {
            let results = retriever.search_sparse(black_box(&query_terms), top_k);
            black_box(results)
        })
    });

    group.bench_function("hybrid_rrf", |b| {
        b.iter(|| {
            let results = retriever.search_hybrid(
                black_box(&query_embedding),
                black_box(&query_terms),
                top_k,
                0.5,
                60,
            );
            black_box(results)
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: FUSION STRATEGIES
// =============================================================================

fn bench_fusion_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_fusion_strategies");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;
    let top_k = 10;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);
    let query_embedding = generate_embedding(999999, dim);
    let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

    // Pre-compute individual search results
    let dense_results = retriever.search_dense(&query_embedding, top_k * 2);
    let sparse_results = retriever.search_sparse(&query_terms, top_k * 2);

    // RRF with different k values
    let rrf_k_values = vec![10, 30, 60, 100];
    for k in rrf_k_values {
        group.bench_with_input(BenchmarkId::new("rrf", k), &k, |b, &k| {
            b.iter(|| {
                let results = retriever.fuse_rrf(
                    black_box(&dense_results),
                    black_box(&sparse_results),
                    top_k,
                    k,
                    0.5,
                );
                black_box(results)
            })
        });
    }

    // Weighted fusion with different alpha values
    let alpha_values = vec![0.0, 0.3, 0.5, 0.7, 1.0];
    for alpha in alpha_values {
        group.bench_with_input(
            BenchmarkId::new("weighted", format!("{:.1}", alpha)),
            &alpha,
            |b, &alpha| {
                b.iter(|| {
                    let results = retriever.fuse_weighted(
                        black_box(&dense_results),
                        black_box(&sparse_results),
                        top_k,
                        alpha,
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: CORPUS SIZE SCALING
// =============================================================================

fn bench_hybrid_corpus_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_corpus_scale");
    group.measurement_time(Duration::from_secs(15));

    let dim = 384;
    let vocab_size = 10_000;
    let top_k = 10;
    let corpus_sizes = vec![100, 1_000, 5_000, 10_000];

    for size in corpus_sizes {
        let retriever = create_test_retriever(size, dim, vocab_size);
        let query_embedding = generate_embedding(999999, dim);
        let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("hybrid_rrf", size), &size, |b, _| {
            b.iter(|| {
                let results = retriever.search_hybrid(
                    black_box(&query_embedding),
                    black_box(&query_terms),
                    top_k,
                    0.5,
                    60,
                );
                black_box(results)
            })
        });
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: ALPHA TUNING
// =============================================================================

fn bench_alpha_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_alpha_tuning");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;
    let top_k = 10;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);
    let query_embedding = generate_embedding(999999, dim);
    let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

    let alpha_values = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    for alpha in alpha_values {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.1}", alpha)),
            &alpha,
            |b, &alpha| {
                b.iter(|| {
                    let results = retriever.search_hybrid(
                        black_box(&query_embedding),
                        black_box(&query_terms),
                        top_k,
                        alpha,
                        60,
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: QUERY COMPLEXITY
// =============================================================================

fn bench_query_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_query_complexity");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;
    let top_k = 10;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);
    let query_embedding = generate_embedding(999999, dim);

    let term_counts = vec![1, 3, 5, 10, 20];

    for num_terms in term_counts {
        let query_terms: Vec<String> = (0..num_terms).map(|i| format!("term_{}", i * 50)).collect();

        group.throughput(Throughput::Elements(num_terms as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_terms),
            &num_terms,
            |b, _| {
                b.iter(|| {
                    let results = retriever.search_hybrid(
                        black_box(&query_embedding),
                        black_box(&query_terms),
                        top_k,
                        0.5,
                        60,
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: RESULT SET SIZE
// =============================================================================

fn bench_result_set_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_result_size");
    group.measurement_time(Duration::from_secs(10));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);
    let query_embedding = generate_embedding(999999, dim);
    let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

    let topk_values = vec![5, 10, 20, 50, 100];

    for top_k in topk_values {
        group.throughput(Throughput::Elements(top_k as u64));
        group.bench_with_input(BenchmarkId::from_parameter(top_k), &top_k, |b, &k| {
            b.iter(|| {
                let results = retriever.search_hybrid(
                    black_box(&query_embedding),
                    black_box(&query_terms),
                    k,
                    0.5,
                    60,
                );
                black_box(results)
            })
        });
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: BATCH HYBRID SEARCH
// =============================================================================

fn bench_batch_hybrid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_batch_search");
    group.measurement_time(Duration::from_secs(15));

    let dim = 384;
    let vocab_size = 10_000;
    let corpus_size = 10_000;
    let top_k = 10;

    let retriever = create_test_retriever(corpus_size, dim, vocab_size);

    let batch_sizes = vec![1, 5, 10, 20];

    for batch_size in batch_sizes {
        let queries: Vec<(Vec<f32>, Vec<String>)> = (0..batch_size)
            .map(|i| {
                let embedding = generate_embedding(900000 + i, dim);
                let terms: Vec<String> = (0..5).map(|j| format!("term_{}", (i + j) * 50)).collect();
                (embedding, terms)
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<Vec<(usize, f32)>> = queries
                        .iter()
                        .map(|(emb, terms)| retriever.search_hybrid(emb, terms, top_k, 0.5, 60))
                        .collect();
                    black_box(results)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &batch_size,
            |b, _| {
                use rayon::prelude::*;
                b.iter(|| {
                    let results: Vec<Vec<(usize, f32)>> = queries
                        .par_iter()
                        .map(|(emb, terms)| retriever.search_hybrid(emb, terms, top_k, 0.5, 60))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: INVERTED INDEX OPERATIONS
// =============================================================================

fn bench_inverted_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_inverted_index");
    group.measurement_time(Duration::from_secs(10));

    let vocab_size = 10_000;
    let corpus_sizes = vec![1_000, 5_000, 10_000];

    for corpus_size in corpus_sizes {
        // Build inverted index
        let mut inverted_index: HashMap<String, Vec<(usize, f32)>> = HashMap::new();
        for i in 0..corpus_size {
            let sparse = generate_sparse_embedding(i, vocab_size);
            for (term, weight) in sparse {
                inverted_index.entry(term).or_default().push((i, weight));
            }
        }

        let query_terms: Vec<String> = (0..5).map(|i| format!("term_{}", i * 100)).collect();

        group.bench_with_input(
            BenchmarkId::new("lookup", corpus_size),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut doc_scores: HashMap<usize, f32> = HashMap::new();
                    for term in &query_terms {
                        if let Some(postings) = inverted_index.get(term) {
                            for (doc_id, weight) in postings {
                                *doc_scores.entry(*doc_id).or_insert(0.0) += weight;
                            }
                        }
                    }
                    black_box(doc_scores)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: OVERLAP HANDLING
// =============================================================================

fn bench_overlap_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_overlap");
    group.measurement_time(Duration::from_secs(10));

    let top_k = 10;
    let rrf_k = 60;

    // Create result sets with varying overlap
    let create_results = |size: usize, offset: usize| -> Vec<(usize, f32)> {
        (0..size)
            .map(|i| (i + offset, 1.0 / (i + 1) as f32))
            .collect()
    };

    // 100% overlap
    let dense_100 = create_results(100, 0);
    let sparse_100 = create_results(100, 0);

    group.bench_function("overlap_100pct", |b| {
        b.iter(|| {
            let mut fused: HashMap<usize, f32> = HashMap::new();
            for (rank, (id, _)) in dense_100.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            for (rank, (id, _)) in sparse_100.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            let mut results: Vec<_> = fused.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            black_box(results)
        })
    });

    // 50% overlap
    let dense_50 = create_results(100, 0);
    let sparse_50 = create_results(100, 50);

    group.bench_function("overlap_50pct", |b| {
        b.iter(|| {
            let mut fused: HashMap<usize, f32> = HashMap::new();
            for (rank, (id, _)) in dense_50.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            for (rank, (id, _)) in sparse_50.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            let mut results: Vec<_> = fused.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            black_box(results)
        })
    });

    // 0% overlap
    let dense_0 = create_results(100, 0);
    let sparse_0 = create_results(100, 100);

    group.bench_function("overlap_0pct", |b| {
        b.iter(|| {
            let mut fused: HashMap<usize, f32> = HashMap::new();
            for (rank, (id, _)) in dense_0.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            for (rank, (id, _)) in sparse_0.iter().enumerate() {
                *fused.entry(*id).or_insert(0.0) += 0.5 / (rrf_k + rank + 1) as f32;
            }
            let mut results: Vec<_> = fused.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            black_box(results)
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = method_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .with_plots();
    targets = bench_search_methods, bench_fusion_strategies
);

criterion_group!(
    name = scale_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3));
    targets = bench_hybrid_corpus_scale, bench_alpha_tuning
);

criterion_group!(
    name = query_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2));
    targets = bench_query_complexity, bench_result_set_size
);

criterion_group!(
    name = advanced_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3));
    targets = bench_batch_hybrid_search, bench_inverted_index, bench_overlap_handling
);

criterion_main!(
    method_benches,
    scale_benches,
    query_benches,
    advanced_benches
);
