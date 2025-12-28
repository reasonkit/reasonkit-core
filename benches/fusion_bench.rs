//! Benchmarks for RRF fusion algorithms
//!
//! Performance target: < 5ms for fusion operations on typical result sets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reasonkit::retrieval::fusion::{to_ranked_results, FusionEngine, RankedResult};
use std::collections::HashMap;
use uuid::Uuid;

/// Generate ranked results for benchmarking
fn generate_ranked_results(count: usize, rank_offset: usize) -> Vec<RankedResult> {
    (0..count)
        .map(|i| RankedResult {
            id: Uuid::from_u128(i as u128),
            score: (count - i) as f32,
            rank: i + rank_offset,
        })
        .collect()
}

/// Benchmark RRF fusion with different result set sizes
fn bench_rrf_fusion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_fusion_scaling");

    let sizes = vec![10, 50, 100, 500, 1000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            let engine = FusionEngine::rrf(60);
            let mut results = HashMap::new();
            results.insert("dense".to_string(), generate_ranked_results(s, 0));
            results.insert("sparse".to_string(), generate_ranked_results(s, 0));

            b.iter(|| {
                let fused = engine.fuse(black_box(results.clone())).unwrap();
                black_box(fused);
            });
        });
    }

    group.finish();
}

/// Benchmark different RRF k values
fn bench_rrf_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_k_values");

    let k_values = vec![10, 30, 60, 100, 200];
    let mut base_results = HashMap::new();
    base_results.insert("dense".to_string(), generate_ranked_results(100, 0));
    base_results.insert("sparse".to_string(), generate_ranked_results(100, 0));

    for k in k_values {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k_val| {
            let engine = FusionEngine::rrf(k_val);

            b.iter(|| {
                let fused = engine.fuse(black_box(base_results.clone())).unwrap();
                black_box(fused);
            });
        });
    }

    group.finish();
}

/// Benchmark weighted sum fusion
fn bench_weighted_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum_fusion");

    let weights = vec![0.0, 0.3, 0.5, 0.7, 1.0];
    let mut base_results = HashMap::new();
    base_results.insert("dense".to_string(), generate_ranked_results(100, 0));
    base_results.insert("sparse".to_string(), generate_ranked_results(100, 0));

    for weight in weights {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.1}", weight)),
            &weight,
            |b, &w| {
                let engine = FusionEngine::weighted(w);

                b.iter(|| {
                    let fused = engine.fuse(black_box(base_results.clone())).unwrap();
                    black_box(fused);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RBF fusion with different rho values
fn bench_rbf_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_fusion");

    let rho_values = vec![0.5, 0.7, 0.8, 0.9, 0.95];
    let mut base_results = HashMap::new();
    base_results.insert("dense".to_string(), generate_ranked_results(100, 0));
    base_results.insert("sparse".to_string(), generate_ranked_results(100, 0));

    for rho in rho_values {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.2}", rho)),
            &rho,
            |b, &r| {
                use reasonkit::retrieval::fusion::FusionStrategy;
                let engine = FusionEngine::new(FusionStrategy::RankBiasedFusion { rho: r });

                b.iter(|| {
                    let fused = engine.fuse(black_box(base_results.clone())).unwrap();
                    black_box(fused);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-method fusion (3+ methods)
fn bench_multi_method_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_method_fusion");

    let method_counts = vec![2, 3, 4, 5];

    for method_count in method_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(method_count),
            &method_count,
            |b, &mc| {
                let engine = FusionEngine::rrf(60);
                let mut results = HashMap::new();

                for i in 0..mc {
                    results.insert(
                        format!("method_{}", i),
                        generate_ranked_results(100, i * 10),
                    );
                }

                b.iter(|| {
                    let fused = engine.fuse(black_box(results.clone())).unwrap();
                    black_box(fused);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark result conversion (scoring to ranking)
fn bench_to_ranked_results(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_ranked_results");

    let sizes = vec![10, 50, 100, 500, 1000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            let scored: Vec<(Uuid, f32)> = (0..s)
                .map(|i| (Uuid::from_u128(i as u128), (s - i) as f32))
                .collect();

            b.iter(|| {
                let ranked = to_ranked_results(black_box(scored.clone()));
                black_box(ranked);
            });
        });
    }

    group.finish();
}

/// Benchmark overlapping vs non-overlapping results
fn bench_overlap_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap_scenarios");

    let engine = FusionEngine::rrf(60);

    // Scenario 1: 100% overlap
    let mut overlap_100 = HashMap::new();
    overlap_100.insert("dense".to_string(), generate_ranked_results(100, 0));
    overlap_100.insert("sparse".to_string(), generate_ranked_results(100, 0));

    group.bench_function("overlap_100", |b| {
        b.iter(|| {
            let fused = engine.fuse(black_box(overlap_100.clone())).unwrap();
            black_box(fused);
        });
    });

    // Scenario 2: 50% overlap
    let mut overlap_50 = HashMap::new();
    let dense_results = generate_ranked_results(100, 0);
    let mut sparse_results = generate_ranked_results(50, 0);
    sparse_results.extend(
        (50..100)
            .map(|i| RankedResult {
                id: Uuid::from_u128((i + 1000) as u128),
                score: (100 - i) as f32,
                rank: i,
            })
            .collect::<Vec<_>>(),
    );
    overlap_50.insert("dense".to_string(), dense_results);
    overlap_50.insert("sparse".to_string(), sparse_results);

    group.bench_function("overlap_50", |b| {
        b.iter(|| {
            let fused = engine.fuse(black_box(overlap_50.clone())).unwrap();
            black_box(fused);
        });
    });

    // Scenario 3: 0% overlap
    let mut overlap_0 = HashMap::new();
    overlap_0.insert("dense".to_string(), generate_ranked_results(100, 0));
    let sparse_results_no_overlap: Vec<RankedResult> = (0..100)
        .map(|i| RankedResult {
            id: Uuid::from_u128((i + 1000) as u128),
            score: (100 - i) as f32,
            rank: i,
        })
        .collect();
    overlap_0.insert("sparse".to_string(), sparse_results_no_overlap);

    group.bench_function("overlap_0", |b| {
        b.iter(|| {
            let fused = engine.fuse(black_box(overlap_0.clone())).unwrap();
            black_box(fused);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rrf_fusion_scaling,
    bench_rrf_k_values,
    bench_weighted_sum,
    bench_rbf_fusion,
    bench_multi_method_fusion,
    bench_to_ranked_results,
    bench_overlap_scenarios,
);

criterion_main!(benches);
