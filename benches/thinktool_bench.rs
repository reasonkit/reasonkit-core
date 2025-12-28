//! ThinkTools Protocol Execution Benchmark
//!
//! Measures performance of the core reasoning protocol execution.
//! Uses mock LLM to isolate protocol orchestration from API latency.
//!
//! Performance target: < 10ms for protocol orchestration (excluding LLM time)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Import from the library (lib.name = "reasonkit" in Cargo.toml)
use reasonkit::thinktool::executor::{ExecutorConfig, ProtocolExecutor, ProtocolInput};

/// Benchmark protocol execution with mock LLM
fn bench_protocol_execution(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Create executor with mock LLM
    let config = ExecutorConfig::mock();
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("thinktool_execution");
    group.measurement_time(Duration::from_secs(10));

    // Test queries of varying complexity
    let queries = vec![
        ("short", "What is 2+2?"),
        ("medium", "Explain the trade-offs between microservices and monoliths."),
        ("long", "Analyze the economic implications of artificial intelligence on the labor market over the next decade, considering automation, job displacement, new job creation, and policy responses."),
    ];

    for (name, query) in queries {
        let input = ProtocolInput::query(query);

        group.bench_with_input(BenchmarkId::new("gigathink", name), &input, |b, input| {
            b.to_async(&rt).iter(|| async {
                let result = executor.execute("gigathink", input.clone()).await;
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("laserlogic", name), &input, |b, input| {
            b.to_async(&rt).iter(|| async {
                let result = executor.execute("laserlogic", input.clone()).await;
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark profile chains (multiple protocols in sequence)
fn bench_profile_chains(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let config = ExecutorConfig::mock();
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("profile_chains");
    group.measurement_time(Duration::from_secs(15));

    let query = "Should startups use microservices or monolith architecture?";
    let input = ProtocolInput::query(query);

    // quick = GigaThink → LaserLogic (2 protocols)
    group.bench_function("quick_profile", |b| {
        b.to_async(&rt).iter(|| async {
            let result = executor.execute_profile("quick", input.clone()).await;
            black_box(result)
        });
    });

    // balanced = GigaThink → LaserLogic → BedRock → ProofGuard (4 protocols)
    group.bench_function("balanced_profile", |b| {
        b.to_async(&rt).iter(|| async {
            let result = executor.execute_profile("balanced", input.clone()).await;
            black_box(result)
        });
    });

    // deep = All 5 protocols
    group.bench_function("deep_profile", |b| {
        b.to_async(&rt).iter(|| async {
            let result = executor.execute_profile("deep", input.clone()).await;
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark protocol step overhead
fn bench_step_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let config = ExecutorConfig::mock();
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("step_overhead");

    // Measure overhead per step by comparing protocols with different step counts
    // GigaThink: 3 steps
    // LaserLogic: 3 steps
    // BedRock: 3 steps
    // ProofGuard: 3 steps
    // BrutalHonesty: 3 steps

    let input = ProtocolInput::query("Test query");

    let protocols = vec![
        ("gigathink", 3),
        ("laserlogic", 3),
        ("bedrock", 3),
        ("proofguard", 3),
        ("brutalhonesty", 3),
    ];

    for (protocol, steps) in protocols {
        group.bench_with_input(
            BenchmarkId::new(protocol, format!("{}_steps", steps)),
            &input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    let result = executor.execute(protocol, input.clone()).await;
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent protocol execution
fn bench_concurrent_execution(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let config = ExecutorConfig::mock();
    let executor = std::sync::Arc::new(ProtocolExecutor::with_config(config).unwrap());

    let mut group = c.benchmark_group("concurrent_execution");
    group.measurement_time(Duration::from_secs(15));

    for concurrency in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_gigathink", concurrency),
            &concurrency,
            |b, &n| {
                b.to_async(&rt).iter(|| async {
                    let executor = executor.clone();
                    let futures: Vec<_> = (0..n)
                        .map(|i| {
                            let exec = executor.clone();
                            let input = ProtocolInput::query(format!("Query {}", i));
                            async move { exec.execute("gigathink", input).await }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_protocol_execution,
    bench_profile_chains,
    bench_step_overhead,
    bench_concurrent_execution,
);
criterion_main!(benches);
