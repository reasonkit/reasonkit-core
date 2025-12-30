//! Benchmarks for document ingestion pipeline
//!
//! Performance target: < 5ms for chunking operations on typical documents

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Generate sample text of varying sizes
fn generate_sample_text(size: usize) -> String {
    let base_text = "This is a sample paragraph about machine learning and artificial intelligence. \
                     It discusses various techniques like chain-of-thought reasoning, self-consistency, \
                     and tree-of-thoughts prompting. These methods improve the performance of large \
                     language models on complex reasoning tasks. ";

    base_text.repeat(size / base_text.len() + 1)[..size].to_string()
}

/// Simple chunking function (fixed-size)
fn chunk_fixed_size(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= chars.len() {
            break;
        }

        start = if overlap < chunk_size {
            end - overlap
        } else {
            end
        };
    }

    chunks
}

/// Semantic chunking (sentence-based)
fn chunk_by_sentences(text: &str, max_chunk_size: usize) -> Vec<String> {
    let sentences: Vec<&str> = text
        .split(|c| ['.', '!', '?'].contains(&c))
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for sentence in sentences {
        let sentence = sentence.trim();
        if current_chunk.len() + sentence.len() > max_chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            current_chunk.clear();
        }
        current_chunk.push_str(sentence);
        current_chunk.push_str(". ");
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Benchmark fixed-size chunking
fn bench_chunk_fixed_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_fixed_size");

    let chunk_sizes = vec![256, 512, 1024, 2048];
    let text = generate_sample_text(10000);

    for chunk_size in chunk_sizes {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &cs| {
                b.iter(|| {
                    let chunks = chunk_fixed_size(black_box(&text), cs, cs / 4);
                    black_box(chunks);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sentence-based chunking
fn bench_chunk_sentences(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_sentences");

    let text_sizes = vec![1000, 5000, 10000, 50000];

    for size in text_sizes {
        let text = generate_sample_text(size);
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let chunks = chunk_by_sentences(black_box(&text), 512);
                black_box(chunks);
            });
        });
    }

    group.finish();
}

/// Benchmark chunking with different overlap ratios
fn bench_chunk_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_overlap");

    let text = generate_sample_text(10000);
    let chunk_size = 512;
    let overlap_ratios = vec![0.0, 0.1, 0.25, 0.5];

    for ratio in overlap_ratios {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}%", ratio * 100.0)),
            &ratio,
            |b, &r| {
                let overlap = (chunk_size as f32 * r) as usize;
                b.iter(|| {
                    let chunks = chunk_fixed_size(black_box(&text), chunk_size, overlap);
                    black_box(chunks);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark text cleaning/normalization
fn bench_text_cleaning(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_cleaning");

    let dirty_text = "This    is   text\n\nwith   irregular\t\tspacing.\n\n\n\nAnd multiple lines.";
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let text = dirty_text.repeat(size / dirty_text.len() + 1)[..size].to_string();
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                // Normalize whitespace
                let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
                black_box(cleaned);
            });
        });
    }

    group.finish();
}

/// Benchmark word counting
fn bench_word_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("word_count");

    let sizes = vec![100, 500, 1000, 5000, 10000];

    for size in sizes {
        let text = generate_sample_text(size);
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let count = text.split_whitespace().count();
                black_box(count);
            });
        });
    }

    group.finish();
}

/// Benchmark metadata extraction (mock)
fn bench_metadata_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_extraction");

    let text = generate_sample_text(5000);

    group.bench_function("extract_title", |b| {
        b.iter(|| {
            // Extract first line/sentence as title
            let title = text.lines().next().unwrap_or("");
            black_box(title);
        });
    });

    group.bench_function("extract_keywords", |b| {
        b.iter(|| {
            // Simple keyword extraction (most common words)
            use std::collections::HashMap;

            let mut word_freq: HashMap<&str, usize> = HashMap::new();
            for word in text.split_whitespace() {
                *word_freq.entry(word).or_insert(0) += 1;
            }

            let mut words: Vec<_> = word_freq.into_iter().collect();
            words.sort_by(|a, b| b.1.cmp(&a.1));
            let keywords: Vec<&str> = words.into_iter().take(10).map(|(w, _)| w).collect();
            black_box(keywords);
        });
    });

    group.finish();
}

/// Benchmark chunk deduplication
fn bench_chunk_deduplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_deduplication");

    let sizes = vec![10, 50, 100, 500];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            // Create chunks with some duplicates
            let chunks: Vec<String> = (0..s).map(|i| format!("Chunk {}", i % (s / 2))).collect();

            b.iter(|| {
                use std::collections::HashSet;
                let mut seen = HashSet::new();
                let unique: Vec<&String> = chunks.iter().filter(|c| seen.insert(*c)).collect();
                black_box(unique);
            });
        });
    }

    group.finish();
}

/// Benchmark parallel chunking
fn bench_parallel_chunking(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_chunking");

    let num_documents = vec![10, 50, 100];

    for num_docs in num_documents {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_docs),
            &num_docs,
            |b, &nd| {
                let documents: Vec<String> = (0..nd).map(|_| generate_sample_text(5000)).collect();

                b.iter(|| {
                    let all_chunks: Vec<Vec<String>> = documents
                        .par_iter()
                        .map(|doc| chunk_fixed_size(doc, 512, 128))
                        .collect();
                    black_box(all_chunks);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hash computation for chunk IDs
fn bench_chunk_hashing(c: &mut Criterion) {
    use sha2::{Digest, Sha256};

    let mut group = c.benchmark_group("chunk_hashing");

    let chunk_sizes = vec![128, 256, 512, 1024, 2048];

    for size in chunk_sizes {
        let chunk = generate_sample_text(size);
        group.throughput(Throughput::Bytes(chunk.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut hasher = Sha256::new();
                hasher.update(black_box(&chunk));
                let hash = hasher.finalize();
                black_box(hash);
            });
        });
    }

    group.finish();
}

/// Benchmark token counting (approximate)
fn bench_token_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_counting");

    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let text = generate_sample_text(size);
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                // Approximate: chars / 4 (GPT-style tokenization approximation)
                let token_count = text.len() / 4;
                black_box(token_count);
            });
        });

        group.bench_with_input(BenchmarkId::new("word_based", size), &size, |b, _| {
            b.iter(|| {
                // Alternative: word count * 1.3
                let token_count = (text.split_whitespace().count() as f32 * 1.3) as usize;
                black_box(token_count);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chunk_fixed_size,
    bench_chunk_sentences,
    bench_chunk_overlap,
    bench_text_cleaning,
    bench_word_count,
    bench_metadata_extraction,
    bench_chunk_deduplication,
    bench_parallel_chunking,
    bench_chunk_hashing,
    bench_token_counting,
);

criterion_main!(benches);
