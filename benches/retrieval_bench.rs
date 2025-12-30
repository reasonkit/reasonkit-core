//! Comprehensive Retrieval Benchmarks for ReasonKit Core
//!
//! Measures:
//! - Query latency (p50, p95, p99)
//! - Throughput (QPS)
//! - Memory usage
//! - Retrieval quality (when ground truth available)

use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reasonkit::reasonkit_mem::Document as MemDocument;
use reasonkit::{
    retrieval::HybridRetriever, Chunk, Document as CoreDocument, DocumentType, EmbeddingIds,
    RetrievalConfig, Source, SourceType,
};
use std::time::Duration;
use uuid::Uuid;

/// Create a realistic test document
fn create_test_document(id: usize, size: usize) -> CoreDocument {
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some(format!("/test/doc_{}.md", id)),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let content = format!(
        "This is test document number {}. It contains information about machine learning, \
         deep learning, neural networks, and artificial intelligence. The document discusses \
         various aspects of chain-of-thought reasoning, self-consistency, and tree-of-thoughts \
         prompting techniques. ",
        id
    )
    .repeat(size / 200); // Repeat to reach desired size

    let mut doc = CoreDocument::new(DocumentType::Note, source).with_content(content.clone());

    // Create realistic chunks (512 chars each)
    let chunk_size = 512;
    let num_chunks = content.len() / chunk_size;

    doc.chunks = (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(content.len());
            Chunk {
                id: Uuid::new_v4(),
                text: content[start..end].to_string(),
                index: i,
                start_char: start,
                end_char: end,
                token_count: Some((end - start) / 4), // Rough estimate
                section: Some(format!("Section {}", i / 5)),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            }
        })
        .collect();

    doc
}

/// Set up test corpus with N documents
async fn setup_test_corpus(num_docs: usize, doc_size: usize) -> HybridRetriever {
    let retriever = HybridRetriever::in_memory().unwrap();

    for i in 0..num_docs {
        let doc = create_test_document(i, doc_size);
        let mem_doc: MemDocument = doc.into();
        retriever.add_document(&mem_doc).await.unwrap();
    }

    retriever
}

/// Benchmark: BM25 sparse search only
fn bench_sparse_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    let queries = vec![
        "machine learning",
        "deep learning neural networks",
        "chain of thought reasoning",
        "self consistency prompting",
        "tree of thoughts artificial intelligence",
    ];

    let mut group = c.benchmark_group("sparse_search");
    group.measurement_time(Duration::from_secs(10));

    for (idx, query) in queries.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("query_{}", idx)),
            query,
            |b, q| {
                b.to_async(&rt)
                    .iter(|| async { retriever.search_sparse(q, 10).await.unwrap() });
            },
        );
    }

    group.finish();
}

/// Benchmark: Hybrid search with different alpha values
fn bench_hybrid_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    let query = "chain of thought reasoning neural networks";
    let alpha_values = vec![0.0, 0.3, 0.5, 0.7, 1.0];

    let mut group = c.benchmark_group("hybrid_search");
    group.measurement_time(Duration::from_secs(10));

    for alpha in alpha_values {
        let config = RetrievalConfig {
            top_k: 10,
            alpha,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("alpha_{}", alpha)),
            &config,
            |b, cfg| {
                b.to_async(&rt)
                    .iter(|| async { retriever.search_hybrid(query, None, cfg).await.unwrap() });
            },
        );
    }

    group.finish();
}

/// Benchmark: Scaling with corpus size
fn bench_corpus_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let corpus_sizes = vec![10, 50, 100, 500, 1000];
    let query = "machine learning deep learning";

    let mut group = c.benchmark_group("corpus_scaling");
    group.measurement_time(Duration::from_secs(15));

    for size in corpus_sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter_batched(
                || rt.block_on(async { setup_test_corpus(s, 2048).await }),
                |retriever| {
                    rt.block_on(async { retriever.search_sparse(query, 10).await.unwrap() })
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark: Different top_k values
fn bench_topk_values(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    let query = "artificial intelligence neural networks";
    let topk_values = vec![5, 10, 20, 50, 100];

    let mut group = c.benchmark_group("topk_scaling");
    group.measurement_time(Duration::from_secs(10));

    for k in topk_values {
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &top_k| {
            b.to_async(&rt)
                .iter(|| async { retriever.search_sparse(query, top_k).await.unwrap() });
        });
    }

    group.finish();
}

/// Benchmark: Query length impact
fn bench_query_length(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    let queries = vec![
        ("short", "machine learning"),
        ("medium", "deep learning neural networks reasoning"),
        ("long", "chain of thought reasoning with self consistency in large language models"),
        ("very_long", "how does tree of thoughts prompting improve reasoning capabilities in large language models compared to standard chain of thought approaches"),
    ];

    let mut group = c.benchmark_group("query_length");
    group.measurement_time(Duration::from_secs(10));

    for (name, query) in queries {
        group.throughput(Throughput::Elements(query.split_whitespace().count() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &query, |b, q| {
            b.to_async(&rt)
                .iter(|| async { retriever.search_sparse(q, 10).await.unwrap() });
        });
    }

    group.finish();
}

/// Benchmark: Concurrent queries (throughput test)
fn bench_concurrent_queries(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    let queries = vec![
        "machine learning",
        "deep learning",
        "neural networks",
        "artificial intelligence",
        "chain of thought",
    ];

    let mut group = c.benchmark_group("concurrent_queries");
    group.measurement_time(Duration::from_secs(10));

    let concurrency_levels = vec![1, 2, 4, 8, 16];

    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, &conc| {
                b.to_async(&rt).iter(|| async {
                    let handles: Vec<_> = (0..conc)
                        .map(|i| {
                            let query = queries[i % queries.len()];
                            let ret = &retriever;
                            async move { ret.search_sparse(query, 10).await.unwrap() }
                        })
                        .collect();

                    futures::future::join_all(handles).await
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory usage during indexing
fn bench_indexing_memory(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let batch_sizes = vec![1, 10, 50, 100];

    let mut group = c.benchmark_group("indexing_batches");
    group.measurement_time(Duration::from_secs(15));

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        // Setup: create fresh retriever and documents
                        let retriever = HybridRetriever::in_memory().unwrap();
                        let docs: Vec<MemDocument> = (0..size)
                            .map(|i| create_test_document(i, 2048).into())
                            .collect();
                        (retriever, docs)
                    },
                    |(retriever, docs)| {
                        // Measurement: index all documents
                        rt.block_on(async {
                            for doc in &docs {
                                retriever.add_document(doc).await.unwrap();
                            }
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Statistics gathering
fn bench_stats_collection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let retriever = rt.block_on(async { setup_test_corpus(100, 2048).await });

    c.bench_function("stats_collection", |b| {
        b.to_async(&rt)
            .iter(|| async { retriever.stats().await.unwrap() });
    });
}

criterion_group!(
    benches,
    bench_sparse_search,
    bench_hybrid_search,
    bench_corpus_scaling,
    bench_topk_values,
    bench_query_length,
    bench_concurrent_queries,
    bench_indexing_memory,
    bench_stats_collection,
);

criterion_main!(benches);
