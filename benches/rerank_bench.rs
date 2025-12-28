//! Cross-Encoder Reranking Benchmarks
//!
//! Measures:
//! - Reranking latency for different candidate set sizes
//! - Throughput (candidates/second)
//! - Score computation overhead
//!
//! Target: < 200ms for top-20 candidates (per DEVELOPMENT_PLAN.md)
//!
//! Research: arXiv:2010.06467 - Cross-Encoders for Document Reranking

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reasonkit::retrieval::{Reranker, RerankerCandidate, RerankerConfig};
use std::time::Duration;
use uuid::Uuid;

/// Create test candidates with varying text lengths
fn create_test_candidates(count: usize) -> Vec<RerankerCandidate> {
    (0..count)
        .map(|i| {
            let text = match i % 5 {
                0 => format!("Machine learning is a subset of artificial intelligence that enables systems to learn from data. Document number {}.", i),
                1 => format!("Deep learning uses neural networks with many layers for pattern recognition and feature extraction. Document {}.", i),
                2 => format!("Natural language processing enables computers to understand and generate human language. Doc {}.", i),
                3 => format!("The weather today is sunny and warm with a high of 75 degrees. Unrelated document {}.", i),
                _ => format!("Computer vision systems can identify objects, faces, and scenes in images and video. Document {}.", i),
            };

            RerankerCandidate {
                id: Uuid::new_v4(),
                text,
                original_score: 1.0 - (i as f32 * 0.01),
                original_rank: i,
            }
        })
        .collect()
}

/// Benchmark: Reranking with different candidate counts
fn bench_rerank_candidate_counts(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = RerankerConfig::default();
    let reranker = Reranker::new(config);
    let query = "machine learning artificial intelligence neural networks";

    let mut group = c.benchmark_group("rerank_candidates");
    group.measurement_time(Duration::from_secs(10));

    for size in [10, 20, 50, 100, 200] {
        let candidates = create_test_candidates(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &candidates,
            |b, candidates| {
                b.to_async(&rt)
                    .iter(|| async { reranker.rerank(query, candidates, 10).await.unwrap() });
            },
        );
    }

    group.finish();
}

/// Benchmark: Reranking top-k selection
fn bench_rerank_top_k(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = RerankerConfig::default();
    let reranker = Reranker::new(config);
    let query = "machine learning";
    let candidates = create_test_candidates(50);

    let mut group = c.benchmark_group("rerank_top_k");
    group.measurement_time(Duration::from_secs(5));

    for k in [1, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.to_async(&rt)
                .iter(|| async { reranker.rerank(query, &candidates, k).await.unwrap() });
        });
    }

    group.finish();
}

/// Benchmark: Compare config presets
fn bench_rerank_configs(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let query = "neural network deep learning";
    let candidates = create_test_candidates(20);

    let mut group = c.benchmark_group("rerank_configs");
    group.measurement_time(Duration::from_secs(5));

    // Default config
    let default_reranker = Reranker::new(RerankerConfig::default());
    group.bench_function("default", |b| {
        b.to_async(&rt).iter(|| async {
            default_reranker
                .rerank(query, &candidates, 10)
                .await
                .unwrap()
        });
    });

    // Fast config
    let fast_reranker = Reranker::new(RerankerConfig::fast());
    group.bench_function("fast", |b| {
        b.to_async(&rt)
            .iter(|| async { fast_reranker.rerank(query, &candidates, 10).await.unwrap() });
    });

    // With score threshold
    let mut threshold_config = RerankerConfig::default();
    threshold_config.score_threshold = Some(0.3);
    let threshold_reranker = Reranker::new(threshold_config);
    group.bench_function("with_threshold", |b| {
        b.to_async(&rt).iter(|| async {
            threshold_reranker
                .rerank(query, &candidates, 10)
                .await
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark: Batched reranking for large candidate sets
fn bench_rerank_batched(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = RerankerConfig::default();
    let reranker = Reranker::new(config);
    let query = "artificial intelligence";
    let candidates = create_test_candidates(100);

    let mut group = c.benchmark_group("rerank_batched");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("unbatched_100", |b| {
        b.to_async(&rt)
            .iter(|| async { reranker.rerank(query, &candidates, 10).await.unwrap() });
    });

    group.bench_function("batched_100", |b| {
        b.to_async(&rt).iter(|| async {
            reranker
                .rerank_batched(query, &candidates, 10)
                .await
                .unwrap()
        });
    });

    group.finish();
}

/// Target latency check: <200ms for 20 candidates
/// This is a hard requirement from DEVELOPMENT_PLAN.md
fn bench_target_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = RerankerConfig::default();
    let reranker = Reranker::new(config);
    let query = "machine learning neural networks deep learning";
    let candidates = create_test_candidates(20);

    let mut group = c.benchmark_group("rerank_target_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    // This benchmark validates the <200ms target
    group.bench_function("20_candidates_target_200ms", |b| {
        b.to_async(&rt)
            .iter(|| async { reranker.rerank(query, &candidates, 20).await.unwrap() });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rerank_candidate_counts,
    bench_rerank_top_k,
    bench_rerank_configs,
    bench_rerank_batched,
    bench_target_latency,
);

criterion_main!(benches);
