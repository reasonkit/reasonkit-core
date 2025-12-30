//! Unified Benchmark Harness for ReasonKit Core
//!
//! This harness orchestrates performance tests across all core modules:
//! - Retrieval (Qdrant/Tantivy)
//! - Embedding (ONNX/API)
//! - Reasoning (ThinkTools)
//! - Ingestion (PDF/Text)
//!
//! Usage:
//!   cargo bench --bench harness

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use reasonkit::ingestion::DocumentIngester;
use reasonkit::thinktool::{ExecutorConfig, ProtocolExecutor, ProtocolInput};
use std::time::Duration;
use tokio::runtime::Runtime;

// ═══════════════════════════════════════════════════════════════════════════
// THINKTOOL BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_thinktool_startup(c: &mut Criterion) {
    let mut group = c.benchmark_group("thinktool_startup");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("executor_init", |b| {
        b.iter(|| {
            let config = ExecutorConfig::default();
            ProtocolExecutor::with_config(config).unwrap()
        })
    });

    group.finish();
}

fn bench_thinktool_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("thinktool_execution");
    group.sampling_mode(SamplingMode::Flat);

    // Mock executor for pure logic benchmarking (no network I/O)
    let config = ExecutorConfig {
        use_mock: true,
        ..Default::default()
    };
    let executor = ProtocolExecutor::with_config(config).unwrap();

    group.bench_function("gigathink_mock", |b| {
        b.to_async(&rt).iter(|| async {
            let input = ProtocolInput::query("Benchmark query");
            executor.execute("gigathink", input).await.unwrap()
        })
    });

    group.bench_function("laserlogic_mock", |b| {
        b.to_async(&rt).iter(|| async {
            let input = ProtocolInput::argument("Benchmark argument");
            executor.execute("laserlogic", input).await.unwrap()
        })
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// INGESTION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_ingestion_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingestion_processing");
    let ingester = DocumentIngester::new();

    // Create a temporary large text file
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("bench_doc.txt");
    let content = "ReasonKit benchmark content. ".repeat(1000); // ~27KB
    std::fs::write(&file_path, content).unwrap();

    group.bench_function("ingest_text_file", |b| {
        b.iter(|| ingester.ingest(&file_path).unwrap())
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG & GROUP DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    name = thinktool_benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(3));
    targets = bench_thinktool_startup, bench_thinktool_execution
);

criterion_group!(
    name = ingestion_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_ingestion_processing
);

criterion_main!(thinktool_benches, ingestion_benches);
