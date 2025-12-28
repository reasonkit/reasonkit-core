# ReasonKit Performance Optimization Guide

> **Target:** <5ms overhead for core operations | <10% token overhead for simple queries
> **Philosophy:** Zero-cost abstractions, minimize allocations, measure everything

---

## Table of Contents

1. [Performance Architecture](#1-performance-architecture)
2. [Benchmarking Infrastructure](#2-benchmarking-infrastructure)
3. [Optimization Techniques](#3-optimization-techniques)
4. [Token Efficiency](#4-token-efficiency)
5. [Latency Optimization](#5-latency-optimization)
6. [Caching Strategy](#6-caching-strategy)
7. [Resource Limits](#7-resource-limits)
8. [Performance Testing](#8-performance-testing)
9. [Optimization Checklist](#9-optimization-checklist)
10. [User-Facing Performance](#10-user-facing-performance)

---

## 1. Performance Architecture

### Design Principles

ReasonKit follows Rust's zero-cost abstraction philosophy:

```
PRINCIPLE                     IMPLEMENTATION
----------------------------------------------------------------------
Zero-cost abstractions   ->   Compile-time dispatch, no runtime vtables
Minimize allocations     ->   Arena allocation, buffer reuse, Cow<str>
Lazy evaluation          ->   On-demand protocol loading, deferred parsing
Efficient serialization  ->   serde with skip_serializing_if, compact JSON
```

### Critical Paths (Performance-Sensitive Code)

| Critical Path | Location | Target Latency | Notes |
|---------------|----------|----------------|-------|
| Protocol parsing | `src/thinktool/protocol.rs` | <1ms | YAML parsing + validation |
| LLM request formatting | `src/thinktool/llm.rs` | <2ms | Prompt construction |
| Response processing | `src/thinktool/step.rs` | <5ms | Parsing + trace update |
| Trace storage | `src/thinktool/trace.rs` | <3ms | Serialize + write |
| BM25 search | `src/retrieval/mod.rs` | <10ms | Tantivy query execution |
| Hybrid fusion | `src/retrieval/fusion.rs` | <5ms | Score normalization |

### Memory Layout Considerations

```rust
// GOOD: Compact enum (1 byte discriminant + max variant size)
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed(String),  // String only allocated when Failed
}

// GOOD: Use Cow<str> for potentially borrowed strings
pub struct ProtocolStep {
    pub id: Cow<'static, str>,      // Often static, avoid allocation
    pub prompt_template: String,     // Usually dynamic, must own
}

// GOOD: Use SmallVec for small, bounded collections
use smallvec::SmallVec;
pub struct StepResult {
    items: SmallVec<[ListItem; 8]>,  // Stack-allocated for <= 8 items
}
```

### Release Profile Configuration

```toml
# Cargo.toml - Already configured in reasonkit-core
[profile.release]
lto = true           # Link-Time Optimization (cross-crate inlining)
codegen-units = 1    # Single codegen unit for better optimization
opt-level = 3        # Maximum optimization
strip = true         # Strip debug symbols for smaller binary

[profile.bench]
lto = true           # LTO for accurate benchmarks
codegen-units = 1    # Consistent with release
```

---

## 2. Benchmarking Infrastructure

### Criterion Benchmarks

ReasonKit uses Criterion.rs for statistically rigorous benchmarking.

#### Existing Benchmark Files

| Benchmark | File | What It Measures |
|-----------|------|------------------|
| `retrieval_bench` | `benches/retrieval_bench.rs` | BM25, hybrid search, corpus scaling |
| `thinktool_bench` | `benches/thinktool_bench.rs` | Protocol execution, profile chains |
| `fusion_bench` | `benches/fusion_bench.rs` | Score fusion algorithms |
| `embedding_bench` | `benches/embedding_bench.rs` | Embedding generation/lookup |
| `raptor_bench` | `benches/raptor_bench.rs` | RAPTOR tree operations |
| `ingestion_bench` | `benches/ingestion_bench.rs` | Document ingestion pipeline |
| `rerank_bench` | `benches/rerank_bench.rs` | Re-ranking algorithms |

#### Adding New Benchmarks

```rust
// benches/new_feature_bench.rs
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use reasonkit::your_module::YourFeature;

fn bench_your_feature(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("your_feature");
    group.measurement_time(Duration::from_secs(10));  // Adequate sampling
    group.sample_size(100);  // 100 samples for statistical validity

    // Parameterized benchmark
    for size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("operation", size),
            &size,
            |b, &s| {
                b.to_async(&rt).iter(|| async {
                    let result = YourFeature::new(s).execute().await;
                    black_box(result)  // Prevent dead code elimination
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_your_feature);
criterion_main!(benches);
```

#### Register in Cargo.toml

```toml
[[bench]]
name = "new_feature_bench"
harness = false
```

### Benchmark Commands

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench -- sparse_search
cargo bench -- thinktool_execution

# Save baseline for comparison
cargo bench -- --save-baseline main

# Compare against baseline (regression detection)
cargo bench -- --baseline main

# Generate HTML report
cargo bench -- --verbose
# Open: target/criterion/report/index.html

# Quick benchmark (fewer samples, faster)
cargo bench -- --quick

# Profile a specific benchmark
cargo bench -- --profile-time 30 sparse_search
```

### Profiling Tools

#### CPU Profiling with flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph (requires root on Linux)
sudo cargo flamegraph --bench retrieval_bench -- sparse_search

# View: flamegraph.svg
```

#### Memory Profiling with heaptrack

```bash
# Linux only
sudo apt install heaptrack heaptrack-gui

# Profile memory allocations
heaptrack ./target/release/rk-core think "test query" --mock

# Analyze
heaptrack_gui heaptrack.rk-core.*.gz
```

#### Perf (Linux)

```bash
# Record performance data
perf record -g ./target/release/rk-core think "test" --mock

# Analyze
perf report

# Annotate specific function
perf annotate
```

#### Instruments (macOS)

```bash
# Use Xcode Instruments
xcrun xctrace record --template 'Time Profiler' \
  --launch -- ./target/release/rk-core think "test" --mock
```

#### Valgrind/Cachegrind

```bash
# Cache analysis
valgrind --tool=cachegrind ./target/release/rk-core think "test" --mock

# View results
cg_annotate cachegrind.out.*
```

---

## 3. Optimization Techniques

### Memory Optimization

#### Arena Allocation (for Short-Lived Objects)

```rust
use bumpalo::Bump;

pub fn process_batch<'a>(arena: &'a Bump, items: &[Input]) -> Vec<&'a Output> {
    items.iter()
        .map(|item| {
            // Allocate in arena - freed all at once when arena is dropped
            arena.alloc(process_item(item))
        })
        .collect()
}
```

#### String Interning (for Repeated Strings)

```rust
use std::collections::HashSet;
use std::sync::Arc;

pub struct StringInterner {
    strings: HashSet<Arc<str>>,
}

impl StringInterner {
    pub fn intern(&mut self, s: &str) -> Arc<str> {
        if let Some(existing) = self.strings.get(s) {
            Arc::clone(existing)
        } else {
            let arc: Arc<str> = s.into();
            self.strings.insert(Arc::clone(&arc));
            arc
        }
    }
}
```

#### Cow<str> Usage

```rust
use std::borrow::Cow;

// GOOD: Avoids allocation when using static strings
pub fn get_prompt(protocol: &str) -> Cow<'static, str> {
    match protocol {
        "gigathink" => Cow::Borrowed(include_str!("../prompts/gigathink.txt")),
        other => Cow::Owned(format!("Custom protocol: {}", other)),
    }
}
```

#### Buffer Reuse

```rust
use std::sync::Mutex;

// Thread-local buffer pool
thread_local! {
    static BUFFER_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
}

pub fn get_buffer() -> Vec<u8> {
    BUFFER_POOL.with(|pool| {
        pool.lock().unwrap().pop().unwrap_or_else(|| Vec::with_capacity(4096))
    })
}

pub fn return_buffer(mut buf: Vec<u8>) {
    buf.clear();  // Clear but keep capacity
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.lock().unwrap();
        if pool.len() < 16 {  // Limit pool size
            pool.push(buf);
        }
    });
}
```

### CPU Optimization

#### Inline Critical Paths

```rust
// Force inlining for hot paths
#[inline(always)]
pub fn normalize_score(score: f32, max: f32) -> f32 {
    if max == 0.0 { 0.0 } else { score / max }
}

// Hint for unlikely branches (cold paths)
#[cold]
fn handle_error(e: Error) -> Result<()> {
    tracing::error!("Error: {:?}", e);
    Err(e)
}
```

#### SIMD Where Applicable

```rust
// Example: SIMD-accelerated cosine similarity (for embeddings)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { cosine_similarity_avx2(a, b) };
        }
    }

    // Fallback to scalar
    cosine_similarity_scalar(a, b)
}
```

#### Avoid Unnecessary Copies

```rust
// BAD: Clones the entire string
fn process_bad(text: String) -> String {
    let processed = text.clone();
    processed.to_uppercase()
}

// GOOD: Takes ownership, avoids clone
fn process_good(text: String) -> String {
    text.to_uppercase()
}

// GOOD: Takes reference when only reading
fn analyze(text: &str) -> usize {
    text.chars().count()
}
```

### I/O Optimization

#### Async I/O with Tokio

```rust
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// GOOD: Non-blocking file operations
pub async fn save_trace_async(trace: &ExecutionTrace, path: &Path) -> Result<()> {
    let json = serde_json::to_vec(trace)?;

    let mut file = fs::File::create(path).await?;
    file.write_all(&json).await?;
    file.sync_all().await?;

    Ok(())
}
```

#### Connection Pooling

```rust
use reqwest::Client;
use std::time::Duration;

// GOOD: Reuse HTTP client with connection pooling
pub fn create_http_client() -> Client {
    Client::builder()
        .pool_max_idle_per_host(10)    // Keep 10 idle connections
        .pool_idle_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(60))
        .build()
        .expect("Failed to create HTTP client")
}
```

#### Request Batching

```rust
// BAD: Sequential API calls
async fn embed_documents_bad(docs: &[String]) -> Vec<Vec<f32>> {
    let mut results = Vec::new();
    for doc in docs {
        results.push(embed_single(doc).await?);
    }
    results
}

// GOOD: Batch API call
async fn embed_documents_good(docs: &[String]) -> Vec<Vec<f32>> {
    // Single API call with multiple inputs
    embed_batch(docs).await?
}

// GOOD: Parallel with bounded concurrency
use futures::stream::{self, StreamExt};

async fn embed_documents_parallel(docs: &[String], concurrency: usize) -> Vec<Vec<f32>> {
    stream::iter(docs)
        .map(|doc| embed_single(doc))
        .buffer_unordered(concurrency)  // Max N concurrent requests
        .collect()
        .await
}
```

#### Response Compression

```rust
// Enable gzip compression in reqwest
let client = Client::builder()
    .gzip(true)
    .brotli(true)  // Also enable Brotli
    .build()?;
```

---

## 4. Token Efficiency

### Prompt Optimization

#### Minimal System Prompts

```rust
// BAD: Verbose system prompt (wastes tokens)
const SYSTEM_PROMPT_BAD: &str = r#"
You are an advanced AI assistant designed to help users with a wide
variety of tasks. You should always be helpful, harmless, and honest.
Please think carefully before responding and consider multiple perspectives.
If you're unsure about something, please say so.
..."#;  // 50+ tokens of boilerplate

// GOOD: Compact system prompt
const SYSTEM_PROMPT_GOOD: &str = "Respond concisely. Cite sources.";  // 5 tokens
```

#### Efficient Formatting

```rust
// BAD: Verbose JSON prompt
let prompt_bad = format!(r#"
Please analyze the following query and provide your response in JSON format:

Query: {}

Your response should include the following fields:
- analysis: Your analysis of the query
- confidence: A number from 0 to 1
- sources: An array of sources you consulted
"#, query);

// GOOD: Compact JSON prompt
let prompt_good = format!(
    "Analyze: \"{}\"\nRespond: {{\"analysis\":..., \"confidence\":0-1, \"sources\":[]}}",
    query
);
```

#### Template Caching

```rust
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Pre-compiled templates (loaded once at startup)
static TEMPLATES: Lazy<HashMap<&str, String>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gigathink", include_str!("templates/gigathink.txt").to_string());
    m.insert("laserlogic", include_str!("templates/laserlogic.txt").to_string());
    m.insert("bedrock", include_str!("templates/bedrock.txt").to_string());
    m
});

pub fn get_template(protocol: &str) -> Option<&'static String> {
    TEMPLATES.get(protocol)
}
```

#### Dynamic Prompts Based on Need

```rust
pub struct PromptBuilder {
    base: String,
    include_examples: bool,
    include_context: bool,
}

impl PromptBuilder {
    pub fn build(&self, query: &str, context: Option<&str>) -> String {
        let mut prompt = self.base.clone();

        // Only include context if provided and enabled
        if self.include_context {
            if let Some(ctx) = context {
                prompt.push_str("\n\nContext: ");
                prompt.push_str(ctx);
            }
        }

        // Only include examples for complex queries
        if self.include_examples && query.len() > 100 {
            prompt.push_str("\n\nExample format:\n...");
        }

        prompt.push_str("\n\nQuery: ");
        prompt.push_str(query);
        prompt
    }
}
```

### Token Measurement

```bash
# Analyze token usage for a query
rk-core analyze-tokens "Your query here"

# Output:
# Query tokens:       12
# System prompt:      25
# ReasonKit overhead: 45
# Total:              82
# Overhead %:         54% (target: <20% for complex queries)
```

#### Token Counting Implementation

```rust
// Using tiktoken-rs for accurate token counting
use tiktoken_rs::cl100k_base;

pub struct TokenAnalyzer {
    bpe: tiktoken_rs::CoreBPE,
}

impl TokenAnalyzer {
    pub fn new() -> Self {
        Self {
            bpe: cl100k_base().unwrap(),
        }
    }

    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    pub fn analyze(&self, query: &str, system_prompt: &str, full_prompt: &str) -> TokenAnalysis {
        let query_tokens = self.count(query);
        let system_tokens = self.count(system_prompt);
        let total_tokens = self.count(full_prompt);
        let overhead_tokens = total_tokens - query_tokens;

        TokenAnalysis {
            query_tokens,
            system_tokens,
            overhead_tokens,
            total_tokens,
            overhead_pct: (overhead_tokens as f64 / total_tokens as f64) * 100.0,
        }
    }
}

pub struct TokenAnalysis {
    pub query_tokens: usize,
    pub system_tokens: usize,
    pub overhead_tokens: usize,
    pub total_tokens: usize,
    pub overhead_pct: f64,
}
```

### Token Efficiency Targets

| Query Type | Target Overhead | Maximum Overhead |
|------------|-----------------|------------------|
| Simple (< 20 tokens) | <10% | 20% |
| Medium (20-100 tokens) | <15% | 25% |
| Complex (> 100 tokens) | <20% | 30% |
| With context (RAG) | <25% | 35% |

---

## 5. Latency Optimization

### Targets by Operation

| Operation | Target | Max Acceptable | Notes |
|-----------|--------|----------------|-------|
| CLI startup | <50ms | 100ms | Cold start, no config |
| Protocol parsing | <1ms | 5ms | YAML parse + validate |
| Request formatting | <5ms | 10ms | Prompt construction |
| Response processing | <10ms | 20ms | Parse + trace update |
| Trace serialization | <3ms | 10ms | JSON serialize |
| BM25 search (1K docs) | <10ms | 50ms | Tantivy query |
| Hybrid search (1K docs) | <20ms | 100ms | BM25 + vector + fusion |
| Embedding lookup | <5ms | 20ms | Qdrant query |

### Optimization Strategies

#### Lazy Initialization

```rust
use once_cell::sync::OnceCell;

// GOOD: Protocol registry loaded only when first needed
static REGISTRY: OnceCell<ProtocolRegistry> = OnceCell::new();

pub fn get_registry() -> &'static ProtocolRegistry {
    REGISTRY.get_or_init(|| {
        ProtocolRegistry::load_default().expect("Failed to load protocols")
    })
}
```

#### Parallel Processing

```rust
use rayon::prelude::*;

// GOOD: Parallel document processing
pub fn process_documents(docs: &[Document]) -> Vec<ProcessedDoc> {
    docs.par_iter()  // Parallel iterator
        .map(|doc| process_single(doc))
        .collect()
}
```

#### Streaming Responses

```rust
use tokio::io::AsyncBufReadExt;

// GOOD: Stream response instead of buffering entirely
pub async fn stream_llm_response(request: &LlmRequest) -> impl Stream<Item = String> {
    let response = client.post(url)
        .json(request)
        .send()
        .await?;

    // Stream line by line
    tokio_util::io::ReaderStream::new(response.bytes_stream())
        .lines()
        .filter_map(|line| {
            // Parse SSE and emit tokens
            parse_sse_token(line.ok()?)
        })
}
```

#### Early Termination

```rust
// GOOD: Stop searching when confidence is high enough
pub async fn search_with_early_exit(
    query: &str,
    target_confidence: f64,
) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut best_confidence = 0.0;

    for source in sources {
        let batch = source.search(query).await?;

        for result in batch {
            if result.confidence > best_confidence {
                best_confidence = result.confidence;
            }
            results.push(result);
        }

        // Early exit if we have high confidence
        if best_confidence >= target_confidence {
            break;
        }
    }

    results
}
```

### Startup Time Optimization

```bash
# Measure startup time
hyperfine --warmup 3 './target/release/rk-core --help'

# Expected: <50ms

# If slow, check:
# 1. Static initializers (use Lazy/OnceCell)
# 2. Config file parsing (cache parsed config)
# 3. File I/O during init (defer to first use)
```

---

## 6. Caching Strategy

### What to Cache

| Item | Cache Type | TTL | Invalidation |
|------|------------|-----|--------------|
| Parsed protocols | In-memory (static) | Forever | On reload |
| Compiled templates | In-memory (static) | Forever | On reload |
| LLM responses | Disk (SQLite) | 24h | On query hash change |
| Embedding vectors | Qdrant | Forever | On document update |
| Search results | In-memory LRU | 5min | On index update |
| Token counts | In-memory | Session | Never |

### Cache Implementation

#### In-Memory LRU Cache

```rust
use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;

pub struct ResponseCache {
    cache: Mutex<LruCache<String, CachedResponse>>,
}

impl ResponseCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap()
            )),
        }
    }

    pub fn get(&self, key: &str) -> Option<CachedResponse> {
        self.cache.lock().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: String, value: CachedResponse) {
        self.cache.lock().unwrap().put(key, value);
    }
}

#[derive(Clone)]
pub struct CachedResponse {
    pub response: String,
    pub cached_at: chrono::DateTime<chrono::Utc>,
    pub ttl_secs: u64,
}

impl CachedResponse {
    pub fn is_valid(&self) -> bool {
        let elapsed = chrono::Utc::now()
            .signed_duration_since(self.cached_at)
            .num_seconds() as u64;
        elapsed < self.ttl_secs
    }
}
```

#### Disk Cache (SQLite)

```rust
use rusqlite::{Connection, params};

pub struct DiskCache {
    conn: Connection,
}

impl DiskCache {
    pub fn new(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at INTEGER,
                ttl_secs INTEGER
            )",
            [],
        )?;

        // Cleanup expired entries periodically
        conn.execute(
            "DELETE FROM cache WHERE created_at + ttl_secs < unixepoch()",
            [],
        )?;

        Ok(Self { conn })
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.conn.query_row(
            "SELECT value FROM cache
             WHERE key = ? AND created_at + ttl_secs > unixepoch()",
            params![key],
            |row| row.get(0),
        ).ok()
    }

    pub fn put(&self, key: &str, value: &[u8], ttl_secs: u64) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at, ttl_secs)
             VALUES (?, ?, unixepoch(), ?)",
            params![key, value, ttl_secs],
        )?;
        Ok(())
    }
}
```

#### Cache Key Generation

```rust
use sha2::{Sha256, Digest};

pub fn cache_key(query: &str, protocol: &str, model: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.as_bytes());
    hasher.update(b"|");
    hasher.update(protocol.as_bytes());
    hasher.update(b"|");
    hasher.update(model.as_bytes());

    format!("{:x}", hasher.finalize())
}
```

### Cache Warming (Enterprise)

```rust
pub async fn warm_cache(common_queries: &[&str]) {
    for query in common_queries {
        // Pre-compute embeddings
        let embedding = embed(query).await;
        EMBEDDING_CACHE.put(query.to_string(), embedding);

        // Pre-compute common protocol outputs with mock
        for protocol in ["gigathink", "laserlogic", "bedrock"] {
            let key = cache_key(query, protocol, "mock");
            if !RESPONSE_CACHE.contains(&key) {
                let result = execute_protocol(protocol, query, true).await;
                RESPONSE_CACHE.put(key, result);
            }
        }
    }
}
```

---

## 7. Resource Limits

### Memory Limits

```rust
/// Resource limit configuration
pub struct ResourceLimits {
    /// Maximum trace size in bytes (default: 10MB)
    pub max_trace_size: usize,

    /// Maximum response buffer in bytes (default: 1MB)
    pub max_response_buffer: usize,

    /// Maximum embedding cache entries (default: 10000)
    pub max_embedding_cache: usize,

    /// Maximum document size for ingestion (default: 50MB)
    pub max_document_size: usize,

    /// Maximum concurrent LLM requests (default: 10)
    pub max_concurrent_requests: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_trace_size: 10 * 1024 * 1024,       // 10 MB
            max_response_buffer: 1 * 1024 * 1024,   // 1 MB
            max_embedding_cache: 10_000,            // 10K embeddings
            max_document_size: 50 * 1024 * 1024,    // 50 MB
            max_concurrent_requests: 10,
        }
    }
}
```

### Concurrency Limits

```rust
use tokio::sync::Semaphore;

/// Rate limiter for LLM requests
pub struct RateLimiter {
    semaphore: Semaphore,
    requests_per_second: f64,
}

impl RateLimiter {
    pub fn new(max_concurrent: usize, rps: f64) -> Self {
        Self {
            semaphore: Semaphore::new(max_concurrent),
            requests_per_second: rps,
        }
    }

    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("Semaphore closed")
    }
}
```

### Backpressure Handling

```rust
use tokio::sync::mpsc;

/// Bounded channel for backpressure
pub fn create_work_queue(capacity: usize) -> (mpsc::Sender<Work>, mpsc::Receiver<Work>) {
    mpsc::channel(capacity)
}

// Producer drops messages if queue is full (or blocks with send().await)
pub async fn submit_work(tx: &mpsc::Sender<Work>, work: Work) -> Result<()> {
    tx.send_timeout(work, Duration::from_secs(5))
        .await
        .map_err(|_| Error::QueueFull)?;
    Ok(())
}
```

---

## 8. Performance Testing

### Load Testing

```bash
# Install oha (Rust-based HTTP load tester)
cargo install oha

# Test API endpoint (if running as server)
oha -z 30s -c 50 http://localhost:8080/api/think

# Output shows:
# - Requests/sec
# - Latency percentiles (p50, p90, p99)
# - Error rate
```

#### Custom Load Test Script

```bash
#!/bin/bash
# scripts/load_test.sh

QUERIES=(
    "What is 2+2?"
    "Explain microservices vs monolith"
    "Analyze the economic impact of AI"
)

CONCURRENCY=${1:-10}
DURATION=${2:-60}

echo "Load test: ${CONCURRENCY} concurrent, ${DURATION}s duration"

start_time=$(date +%s)
count=0
errors=0

while (( $(date +%s) - start_time < DURATION )); do
    for query in "${QUERIES[@]}"; do
        # Run in background
        (./target/release/rk-core think "$query" --mock >/dev/null 2>&1 || ((errors++))) &
        ((count++))

        # Limit concurrency
        while (( $(jobs -r | wc -l) >= CONCURRENCY )); do
            sleep 0.1
        done
    done
done

wait

elapsed=$(($(date +%s) - start_time))
rate=$(echo "scale=2; $count / $elapsed" | bc)

echo "Completed: $count requests in ${elapsed}s"
echo "Rate: $rate req/s"
echo "Errors: $errors"
```

### Stress Testing

```bash
# Large input stress test
yes "This is a test sentence. " | head -c 1000000 | \
    ./target/release/rk-core think --mock -

# Many concurrent requests
seq 1 1000 | xargs -P 50 -I {} \
    ./target/release/rk-core think "Query {}" --mock

# Extended duration test (memory leaks)
timeout 3600 ./scripts/continuous_test.sh
```

### Regression Testing in CI

```yaml
# .github/workflows/bench.yml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run benchmarks
        run: cargo bench -- --save-baseline pr

      - name: Download baseline
        uses: actions/download-artifact@v4
        with:
          name: bench-baseline
          path: target/criterion
        continue-on-error: true

      - name: Compare to baseline
        run: |
          cargo bench -- --baseline main --no-run
          # Fail if >5% regression
          cargo bench -- --baseline main 2>&1 | tee bench.log
          if grep -q "regressed" bench.log; then
            echo "Performance regression detected!"
            exit 1
          fi

      - name: Upload new baseline
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v4
        with:
          name: bench-baseline
          path: target/criterion
```

### Bisect Capability

```bash
# Find which commit caused regression
git bisect start
git bisect bad HEAD
git bisect good v0.1.0

# Automated bisect with benchmark
git bisect run bash -c '
  cargo build --release && \
  cargo bench -- sparse_search --no-run && \
  cargo bench -- sparse_search 2>&1 | grep -q "faster" && exit 0 || exit 1
'
```

---

## 9. Optimization Checklist

### Before Release

- [ ] Run full benchmark suite: `cargo bench`
- [ ] Compare to baseline: `cargo bench -- --baseline main`
- [ ] Profile hot paths: `cargo flamegraph --bench thinktool_bench`
- [ ] Check memory usage: `heaptrack ./target/release/rk-core think "test" --mock`
- [ ] Verify no regressions: Check Criterion HTML report
- [ ] Test startup time: `hyperfine './target/release/rk-core --help'`
- [ ] Run stress tests: Large inputs, many concurrent requests

### PR Review Checklist

- [ ] No unnecessary allocations in hot paths
- [ ] Async operations don't block
- [ ] Caching used where appropriate
- [ ] Resource limits respected
- [ ] No N+1 query patterns
- [ ] Benchmarks added for new features

### Continuous Monitoring

- [ ] Nightly benchmark runs in CI
- [ ] Weekly review of benchmark trends
- [ ] Quarterly deep performance audit
- [ ] Alert on >5% regression

### Performance Review Schedule

| Frequency | Task | Owner |
|-----------|------|-------|
| Per PR | Run `cargo bench` on changed code | Author |
| Nightly | Full benchmark suite in CI | Automation |
| Weekly | Review benchmark trends | Performance lead |
| Monthly | Profile and optimize hot paths | Team |
| Quarterly | Deep audit, external review | External |

---

## 10. User-Facing Performance

### CLI Performance Mode

```bash
# Performance mode: Optimized for speed over features
rk-core think --performance-mode "Your query"

# What performance mode does:
# - Disables trace saving
# - Uses minimal prompts
# - Skips optional validation
# - Single-pass (no self-consistency)
```

#### Implementation

```rust
impl ExecutorConfig {
    pub fn performance_mode() -> Self {
        Self {
            save_traces: false,
            verbose: false,
            budget: BudgetConfig {
                max_tokens: Some(500),  // Limit output
                ..Default::default()
            },
            self_consistency: None,  // No voting
            ..Default::default()
        }
    }
}
```

### Performance Reporting

```bash
# Detailed timing breakdown
rk-core think --timing "Your query"

# Output:
# Timing Breakdown:
# ─────────────────────────────────────
# Protocol loading:    0.8ms
# Prompt construction: 2.1ms
# LLM request:         1,234.5ms  (external)
# Response parsing:    3.2ms
# Trace serialization: 1.5ms
# ─────────────────────────────────────
# ReasonKit overhead:  7.6ms
# Total wall time:     1,242.1ms
# Overhead %:          0.6%
```

#### Implementation

```rust
use std::time::Instant;

pub struct TimingReport {
    pub protocol_load_ms: f64,
    pub prompt_construction_ms: f64,
    pub llm_request_ms: f64,
    pub response_parsing_ms: f64,
    pub trace_serialization_ms: f64,
}

impl TimingReport {
    pub fn reasonkit_overhead(&self) -> f64 {
        self.protocol_load_ms
            + self.prompt_construction_ms
            + self.response_parsing_ms
            + self.trace_serialization_ms
    }

    pub fn total(&self) -> f64 {
        self.reasonkit_overhead() + self.llm_request_ms
    }

    pub fn overhead_pct(&self) -> f64 {
        (self.reasonkit_overhead() / self.total()) * 100.0
    }

    pub fn display(&self) {
        eprintln!("Timing Breakdown:");
        eprintln!("{}", "─".repeat(40));
        eprintln!("Protocol loading:    {:>8.1}ms", self.protocol_load_ms);
        eprintln!("Prompt construction: {:>8.1}ms", self.prompt_construction_ms);
        eprintln!("LLM request:         {:>8.1}ms  (external)", self.llm_request_ms);
        eprintln!("Response parsing:    {:>8.1}ms", self.response_parsing_ms);
        eprintln!("Trace serialization: {:>8.1}ms", self.trace_serialization_ms);
        eprintln!("{}", "─".repeat(40));
        eprintln!("ReasonKit overhead:  {:>8.1}ms", self.reasonkit_overhead());
        eprintln!("Total wall time:     {:>8.1}ms", self.total());
        eprintln!("Overhead %:          {:>8.1}%", self.overhead_pct());
    }
}
```

### Profile-Based Performance

```bash
# Quick mode: 2 protocols, minimal overhead
rk-core think --profile quick "Question"
# Target: <50ms overhead

# Balanced mode: 4 protocols, standard
rk-core think --profile balanced "Question"
# Target: <100ms overhead

# Deep mode: 5 protocols, thorough
rk-core think --profile deep "Question"
# Target: <200ms overhead

# Paranoid mode: All protocols + validation
rk-core think --profile paranoid "Question"
# Target: <500ms overhead
```

---

## Quick Reference

### Essential Commands

```bash
# Run benchmarks
cargo bench

# Profile with flamegraph
sudo cargo flamegraph --bench retrieval_bench

# Check startup time
hyperfine './target/release/rk-core --help'

# Memory profiling (Linux)
heaptrack ./target/release/rk-core think "test" --mock

# Load test
oha -z 30s -c 50 http://localhost:8080/api/think

# Compare benchmarks
cargo bench -- --baseline main
```

### Performance Targets Summary

| Metric | Target | Max |
|--------|--------|-----|
| CLI startup | <50ms | 100ms |
| Protocol overhead | <5ms | 10ms |
| Token overhead (simple) | <10% | 20% |
| Token overhead (complex) | <20% | 30% |
| Search (1K docs) | <20ms | 100ms |

### Files to Profile First

1. `src/thinktool/executor.rs` - Protocol execution
2. `src/thinktool/llm.rs` - LLM request formatting
3. `src/retrieval/mod.rs` - Search operations
4. `src/thinktool/trace.rs` - Trace serialization

---

_ReasonKit Performance Guide v1.0 | Last Updated: December 2025_
_"Zero-cost abstractions. Measure everything. Optimize what matters."_
