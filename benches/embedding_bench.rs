//! Benchmarks for embedding operations
//!
//! Performance target: < 5ms for mock embedding operations (cache lookups)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;

/// Mock embedding generator (simulates fast local embedding)
fn mock_embed(text: &str) -> Vec<f32> {
    // Simulate deterministic embedding based on text hash
    let hash = text.len() as f32;
    (0..384).map(|i| ((hash + i as f32) * 0.01).sin()).collect()
}

/// Mock sparse embedding generator (simulates BM25-style sparse vectors)
fn mock_sparse_embed(text: &str) -> HashMap<usize, f32> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut sparse = HashMap::new();

    for (idx, word) in words.iter().enumerate() {
        let token_id = word.len() * (idx + 1);
        let weight = 1.0 / (idx + 1) as f32;
        sparse.insert(token_id, weight);
    }

    sparse
}

/// Benchmark dense embedding generation
fn bench_dense_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_embedding");

    let texts = vec![
        ("short", "machine learning"),
        (
            "medium",
            "deep learning neural networks for artificial intelligence",
        ),
        (
            "long",
            "Chain-of-thought prompting is a technique that improves the reasoning \
             capabilities of large language models by decomposing complex problems \
             into intermediate steps.",
        ),
        (
            "very_long",
            "The field of artificial intelligence has seen remarkable progress in recent \
             years, with large language models demonstrating unprecedented capabilities \
             in natural language understanding, generation, and reasoning. These models, \
             trained on vast corpora of text data, have revolutionized how we approach \
             tasks ranging from translation to question answering.",
        ),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &t| {
            b.iter(|| {
                let embedding = mock_embed(black_box(t));
                black_box(embedding);
            });
        });
    }

    group.finish();
}

/// Benchmark sparse embedding generation
fn bench_sparse_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_embedding");

    let texts = vec![
        ("short", "machine learning"),
        (
            "medium",
            "deep learning neural networks for artificial intelligence",
        ),
        (
            "long",
            "Chain-of-thought prompting is a technique that improves the reasoning \
             capabilities of large language models by decomposing complex problems \
             into intermediate steps.",
        ),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Elements(text.split_whitespace().count() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &t| {
            b.iter(|| {
                let embedding = mock_sparse_embed(black_box(t));
                black_box(embedding);
            });
        });
    }

    group.finish();
}

/// Benchmark batch embedding
fn bench_batch_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_embedding");

    let batch_sizes = vec![1, 10, 50, 100, 500];
    let text = "This is a sample text for embedding benchmarks";

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &bs| {
                let texts: Vec<&str> = (0..bs).map(|_| text).collect();

                b.iter(|| {
                    let embeddings: Vec<Vec<f32>> =
                        texts.iter().map(|t| mock_embed(black_box(t))).collect();
                    black_box(embeddings);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark embedding cache operations
fn bench_embedding_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_cache");

    let texts = vec![
        "machine learning",
        "deep learning",
        "neural networks",
        "artificial intelligence",
        "natural language processing",
    ];

    // Pre-populate cache
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    for text in &texts {
        cache.insert(text.to_string(), mock_embed(text));
    }

    // Benchmark cache hits
    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let text = texts[2]; // "neural networks"
            let embedding = cache.get(black_box(text)).unwrap();
            black_box(embedding);
        });
    });

    // Benchmark cache misses
    group.bench_function("cache_miss_and_insert", |b| {
        let mut local_cache = cache.clone();
        let new_text = "reinforcement learning";

        b.iter(|| {
            if !local_cache.contains_key(new_text) {
                let embedding = mock_embed(black_box(new_text));
                local_cache.insert(new_text.to_string(), embedding);
            }
            local_cache.remove(new_text); // Reset for next iteration
        });
    });

    group.finish();
}

/// Benchmark cosine similarity calculation
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    let embedding_sizes = vec![128, 256, 384, 512, 768, 1024];

    for size in embedding_sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            let vec_a: Vec<f32> = (0..s).map(|i| (i as f32 * 0.01).sin()).collect();
            let vec_b: Vec<f32> = (0..s).map(|i| (i as f32 * 0.02).cos()).collect();

            b.iter(|| {
                let dot_product: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
                let norm_a: f32 = vec_a.iter().map(|a| a * a).sum::<f32>().sqrt();
                let norm_b: f32 = vec_b.iter().map(|b| b * b).sum::<f32>().sqrt();
                let similarity = dot_product / (norm_a * norm_b);
                black_box(similarity);
            });
        });
    }

    group.finish();
}

/// Benchmark parallel batch embedding
fn bench_parallel_batch_embedding(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_batch_embedding");

    let batch_sizes = vec![10, 50, 100, 500, 1000];
    let texts: Vec<String> = (0..1000)
        .map(|i| format!("Sample text number {} for embedding", i))
        .collect();

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &bs| {
                let batch: Vec<&str> = texts.iter().take(bs).map(|s| s.as_str()).collect();

                b.iter(|| {
                    let embeddings: Vec<Vec<f32>> =
                        batch.par_iter().map(|t| mock_embed(black_box(t))).collect();
                    black_box(embeddings);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark embedding dimension reduction
fn bench_dimension_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_reduction");

    let full_embedding: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();

    let target_dims = vec![128, 256, 384, 512];

    for target_dim in target_dims {
        group.bench_with_input(
            BenchmarkId::from_parameter(target_dim),
            &target_dim,
            |b, &td| {
                b.iter(|| {
                    // Simple truncation (PCA would be more complex)
                    let reduced: Vec<f32> = full_embedding[..td].to_vec();
                    black_box(reduced);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dense_embedding,
    bench_sparse_embedding,
    bench_batch_embedding,
    bench_embedding_cache,
    bench_cosine_similarity,
    bench_parallel_batch_embedding,
    bench_dimension_reduction,
);

criterion_main!(benches);
