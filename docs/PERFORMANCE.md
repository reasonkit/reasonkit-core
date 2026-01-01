# ReasonKit Performance Tuning Guide

> **Target**: All core loops < 5ms (excluding LLM latency)
> **Version**: 1.0.0
> **Last Updated**: 2026-01-01

This guide provides comprehensive performance optimization recommendations for ReasonKit deployments, from development laptops to production clusters.

---

## Table of Contents

1. [Baseline Performance Expectations](#1-baseline-performance-expectations)
2. [Configuration Options](#2-configuration-options)
3. [Memory Optimization](#3-memory-optimization)
4. [Embedding Model Selection](#4-embedding-model-selection)
5. [Batch Processing](#5-batch-processing)
6. [Profiling and Benchmarking](#6-profiling-and-benchmarking)
7. [Production Deployment](#7-production-deployment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Baseline Performance Expectations

### 1.1 Core Operation Targets

All non-LLM operations must complete within the specified targets:

| Operation | Target | Expected Baseline | Notes |
|-----------|--------|-------------------|-------|
| **Retrieval Operations** |
| Sparse search (BM25) @ 100 docs | < 5ms | ~2.5ms | Pure text matching |
| Hybrid search @ 100 docs | < 10ms | ~8.0ms | Sparse + dense fusion |
| Concurrent 8 queries | < 15ms | ~12ms | Parallel execution |
| **Fusion Operations** |
| RRF @ 100 results (2 methods) | < 5ms | ~1.2ms | Reciprocal Rank Fusion |
| Weighted sum @ 100 results | < 5ms | ~1.5ms | Linear combination |
| Multi-method (3 methods) | < 10ms | ~3.5ms | Complex fusion |
| **Embedding Operations** |
| Mock embedding (cache hit) | < 0.1ms | ~0.02ms | In-memory lookup |
| Batch embedding @ 100 | < 5ms | ~4.5ms | Parallel processing |
| Cosine similarity @ 384-dim | < 1ms | ~0.08ms | Vector math |
| **RAPTOR Operations** |
| Tree creation @ 100 leaves | < 50ms | ~8ms | Hierarchical clustering |
| Tree traversal @ 1000 nodes | < 5ms | ~2.5ms | DFS/BFS |
| Node lookup | < 0.1ms | ~0.03ms | HashMap access |
| **Ingestion Operations** |
| Fixed chunking @ 10KB | < 5ms | ~1.8ms | Text splitting |
| Sentence chunking @ 10KB | < 10ms | ~3.2ms | NLP-based splitting |
| Parallel @ 100 docs | < 50ms | ~35ms | Rayon parallelism |
| **ThinkTools Operations** |
| Protocol orchestration | < 5ms | ~1-2ms | Excluding LLM time |
| Template rendering | < 1ms | ~100-500us | String substitution |
| Confidence extraction | < 0.5ms | ~50-200us | Regex parsing |
| Profile lookup | < 0.1ms | ~0.03ms | Registry access |

### 1.2 LLM Latency Considerations

LLM calls are the dominant factor in end-to-end latency:

| Provider | Model | Typical Latency | Notes |
|----------|-------|-----------------|-------|
| OpenAI | gpt-4o | 1-5s | Varies by prompt length |
| Anthropic | claude-sonnet-4 | 1-4s | Consistent performance |
| Anthropic | claude-opus-4 | 2-8s | Higher quality, slower |
| Local | DeepSeek-V3 | 0.5-2s | Self-hosted, GPU required |
| Local | Llama 3.3 70B | 1-3s | Self-hosted, GPU required |

**Key Insight**: ReasonKit's overhead is typically <1% of total execution time. Focus optimization efforts on reducing LLM calls, not on ReasonKit internals.

### 1.3 Variance Reduction Metrics

Structured prompting via ThinkTools reduces output variance:

| Metric | Raw Prompts | Structured Prompts | Improvement |
|--------|-------------|-------------------|-------------|
| Mean Agreement Rate (TARa) | 96.0% | 98.0% | +2.0 pp |
| Inconsistency Rate | 4.0% | 2.0% | -2.0 pp |
| Variance Reduction | - | - | 5.0% |

*Results from Claude Sonnet 4 at temperature=0, N=5 runs per question.*

---

## 2. Configuration Options

### 2.1 Core Configuration (`config/default.toml`)

```toml
[general]
data_dir = "./data"
log_level = "info"  # trace, debug, info, warn, error

[storage]
backend = "qdrant"  # "qdrant" or "local"

[storage.qdrant]
host = "localhost"
port = 6334
grpc_port = 6333
embedded = true           # Use embedded mode (no external server)
collection = "reasonkit_docs"
vector_size = 1024        # Must match embedding model dimensions
distance = "Cosine"       # Cosine, Euclid, or Dot
quantization = true       # 4x memory reduction, ~2% accuracy loss

[storage.local]
documents_path = "./data/documents"
index_path = "./data/indexes"

[embedding]
backend = "api"  # "api" or "local"

[embedding.api]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536
batch_size = 100

[embedding.local]
model_path = "./models/bge-m3-onnx"
device = "cpu"  # or "cuda"

[indexing]
bm25_enabled = true
hnsw_m = 16              # Higher = more accuracy, more memory
hnsw_ef_construction = 200  # Higher = slower indexing, better quality

[processing]
chunk_size = 512         # tokens
chunk_overlap = 50       # tokens
min_chunk_size = 100     # Avoid tiny fragments

[retrieval]
top_k = 10
alpha = 0.7              # 0 = BM25 only, 1 = vector only
min_score = 0.0
rerank = false           # Enable for better quality, higher latency

[raptor]
enabled = false
max_depth = 3
cluster_size = 10
summarizer = "claude-3-haiku"

[server]
host = "127.0.0.1"
port = 8080
cors = true
timeout = 30
```

### 2.2 Performance-Critical Settings

#### Storage Backend Selection

| Setting | Performance Impact | When to Use |
|---------|-------------------|-------------|
| `embedded = true` | Fastest startup, no network overhead | Development, single-node |
| `embedded = false` | Slightly higher latency, scalable | Production, multi-node |
| `quantization = true` | 4x memory reduction, ~2% accuracy loss | Memory-constrained |
| `quantization = false` | Full precision | Maximum accuracy required |

#### HNSW Parameters

```toml
[indexing]
hnsw_m = 16              # Connections per node (8-64)
hnsw_ef_construction = 200  # Build-time beam width (100-500)
```

| hnsw_m | Memory/Vector | Search Speed | Accuracy |
|--------|---------------|--------------|----------|
| 8 | ~256 bytes | Very fast | 95% |
| 16 | ~512 bytes | Fast | 98% |
| 32 | ~1KB | Medium | 99% |
| 64 | ~2KB | Slow | 99.5% |

**Recommendation**: Start with `hnsw_m = 16` for most use cases.

#### Retrieval Alpha Tuning

The `alpha` parameter controls the balance between sparse (BM25) and dense (vector) search:

```
alpha = 0.0  -> 100% BM25 (keyword matching)
alpha = 0.5  -> 50/50 hybrid
alpha = 0.7  -> 70% vector, 30% BM25 (recommended default)
alpha = 1.0  -> 100% vector (semantic only)
```

| Use Case | Recommended Alpha | Rationale |
|----------|-------------------|-----------|
| Technical documentation | 0.5 | Keywords matter |
| General knowledge | 0.7 | Semantic similarity important |
| Code search | 0.3 | Exact matches critical |
| Conversational search | 0.8 | Semantic understanding key |

### 2.3 Environment Variables

```bash
# Core settings
export REASONKIT_DATA_DIR="/path/to/data"
export REASONKIT_LOG_LEVEL="info"

# API keys (never commit these!)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
export VOYAGE_API_KEY="..."

# Qdrant connection (if not embedded)
export QDRANT_HOST="localhost"
export QDRANT_PORT="6334"
export QDRANT_API_KEY="..."

# Performance tuning
export RAYON_NUM_THREADS="8"      # Parallel processing threads
export TOKIO_WORKER_THREADS="4"   # Async runtime threads
```

---

## 3. Memory Optimization

### 3.1 Memory Usage by Component

| Component | Memory Per Unit | Scaling Factor |
|-----------|-----------------|----------------|
| Raw document | ~1x document size | Linear |
| Chunks | ~1.1x raw (overlap) | Linear |
| BM25 index | ~0.2x corpus size | Linear |
| Dense embeddings (f32) | 4 bytes * dim * chunks | Linear |
| Dense embeddings (quantized) | 1 byte * dim * chunks | Linear |
| HNSW graph | ~512 bytes * chunks (m=16) | Linear |
| RAPTOR tree | ~2x leaf nodes | Logarithmic |

### 3.2 Memory Reduction Strategies

#### 1. Enable Quantization (4x reduction)

```toml
[storage.qdrant]
quantization = true  # Scalar quantization to uint8
```

Impact: ~2% accuracy reduction for 4x memory savings.

#### 2. Reduce Embedding Dimensions

```toml
[embedding.api]
model = "text-embedding-3-small"
dimensions = 512  # Down from 1536
```

| Dimensions | Memory/Chunk | Accuracy (relative) |
|------------|--------------|---------------------|
| 1536 | 6KB | 100% |
| 1024 | 4KB | 98% |
| 512 | 2KB | 95% |
| 256 | 1KB | 90% |

#### 3. Increase Chunk Size

```toml
[processing]
chunk_size = 1024  # Up from 512
chunk_overlap = 100
```

Doubles content per chunk, halves total chunks.

#### 4. Disable RAPTOR (if not needed)

```toml
[raptor]
enabled = false
```

RAPTOR trees add ~2x overhead for hierarchical summaries.

### 3.3 Memory Monitoring

```bash
# Check process memory
ps aux | grep rk-core

# Detailed memory breakdown
cat /proc/$(pgrep rk-core)/status | grep -E "VmRSS|VmSize"

# Qdrant collection stats
curl localhost:6333/collections/reasonkit_docs | jq '.result.points_count, .result.vectors_count'
```

### 3.4 Recommended Memory Sizing

| Corpus Size | Chunks (512 tok) | Memory (quantized) | Memory (full) |
|-------------|------------------|-------------------|---------------|
| 1,000 docs | ~10,000 | ~100MB | ~400MB |
| 10,000 docs | ~100,000 | ~1GB | ~4GB |
| 100,000 docs | ~1,000,000 | ~10GB | ~40GB |
| 1,000,000 docs | ~10,000,000 | ~100GB | ~400GB |

---

## 4. Embedding Model Selection

### 4.1 Supported Models

#### Cloud Providers (via API)

| Provider | Model | Dimensions | Cost/1M tokens | Quality |
|----------|-------|------------|----------------|---------|
| OpenAI | text-embedding-3-large | 3072 | $0.13 | Excellent |
| OpenAI | text-embedding-3-small | 1536 | $0.02 | Good |
| Cohere | embed-english-v3.0 | 1024 | $0.10 | Excellent |
| Voyage | voyage-2 | 1024 | $0.10 | Excellent |
| Anthropic | (via Voyage) | 1024 | $0.10 | Excellent |

#### Local Models (via ONNX)

| Model | Dimensions | Memory | Speed (CPU) | Speed (GPU) |
|-------|------------|--------|-------------|-------------|
| BGE-M3 | 1024 | ~2GB | 50ms/text | 5ms/text |
| BGE-Large-EN | 1024 | ~1.5GB | 40ms/text | 4ms/text |
| all-MiniLM-L6 | 384 | ~100MB | 10ms/text | 1ms/text |
| E5-Large-v2 | 1024 | ~1.5GB | 40ms/text | 4ms/text |

### 4.2 Model Selection Guide

```
High accuracy, cost acceptable:
  -> OpenAI text-embedding-3-large (3072-dim)
  -> Cohere embed-english-v3.0 (1024-dim)

Good accuracy, cost-sensitive:
  -> OpenAI text-embedding-3-small (1536-dim)
  -> Local BGE-M3 with GPU

Low latency, self-hosted:
  -> Local all-MiniLM-L6 (384-dim, CPU-friendly)
  -> Local E5-Large-v2 with GPU

Multilingual:
  -> BGE-M3 (best multilingual support)
  -> Cohere embed-multilingual-v3.0
```

### 4.3 Local Embedding Setup

```bash
# Install with local-embeddings feature
cargo build --release --features local-embeddings

# Download BGE-M3 ONNX model
mkdir -p models
curl -L https://huggingface.co/BAAI/bge-m3/resolve/main/onnx/model.onnx \
  -o models/bge-m3-onnx/model.onnx
curl -L https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json \
  -o models/bge-m3-onnx/tokenizer.json
```

Configure in `config/default.toml`:

```toml
[embedding]
backend = "local"

[embedding.local]
model_path = "./models/bge-m3-onnx"
device = "cuda"  # or "cpu"
```

### 4.4 Embedding Cache

ReasonKit automatically caches embeddings to avoid recomputation:

```
Cache location: $REASONKIT_DATA_DIR/embeddings_cache/
Cache key: SHA-256(model_id + text)
Cache format: Binary (4 bytes per dimension)
```

Cache performance:

| Operation | Latency |
|-----------|---------|
| Cache hit | < 0.1ms |
| Cache miss (API) | 100-500ms |
| Cache miss (local) | 5-50ms |

---

## 5. Batch Processing

### 5.1 Ingestion Optimization

#### Single Document Ingestion

```bash
rk-core ingest /path/to/document.md
```

#### Batch Ingestion (Recommended)

```bash
# Process directory recursively
rk-core ingest /path/to/documents/ --recursive

# With parallel processing
rk-core ingest /path/to/documents/ --parallel 8

# Stream from stdin (for pipelines)
find /path -name "*.md" | rk-core ingest --stdin
```

### 5.2 Parallel Processing Configuration

```bash
# Set thread count via environment
export RAYON_NUM_THREADS=8

# Or via CLI
rk-core ingest /path --threads 8
```

Optimal thread count:

| CPU Cores | Recommended Threads | Notes |
|-----------|---------------------|-------|
| 2 | 2 | Limited parallelism |
| 4 | 4 | Good for development |
| 8 | 6-8 | Leave 2 for system |
| 16+ | 12-14 | Diminishing returns |

### 5.3 Batch Embedding Strategies

#### Sequential (Default)

```toml
[embedding.api]
batch_size = 1  # One at a time
```

#### Batched (Recommended)

```toml
[embedding.api]
batch_size = 100  # Batch requests
```

#### Pipeline (Advanced)

```rust
// In custom code
let embedder = BatchEmbedder::new(config)
    .with_batch_size(100)
    .with_concurrent_batches(4);

embedder.embed_streaming(chunks_iter).await?;
```

### 5.4 Ingestion Performance Expectations

| Corpus Size | Documents | Chunks | Ingestion Time |
|-------------|-----------|--------|----------------|
| Small | 100 | 1,000 | ~30s |
| Medium | 1,000 | 10,000 | ~5min |
| Large | 10,000 | 100,000 | ~1hr |
| Enterprise | 100,000 | 1,000,000 | ~10hr |

*Times assume API embeddings with batch_size=100. Local embeddings with GPU are ~5x faster.*

---

## 6. Profiling and Benchmarking

### 6.1 Running Benchmarks

#### Quick Benchmark

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench retrieval_bench
cargo bench --bench fusion_bench
cargo bench --bench embedding_bench
cargo bench --bench thinktool_bench
```

#### With Baseline Comparison

```bash
# Save current performance as baseline
cargo bench -- --save-baseline main

# Compare against baseline after changes
cargo bench -- --baseline main
```

#### Using the Helper Script

```bash
./scripts/run_benchmarks.sh            # Run all
./scripts/run_benchmarks.sh fusion     # Run specific
./scripts/run_benchmarks.sh --baseline # Save baseline
./scripts/run_benchmarks.sh --compare  # Check regressions
```

### 6.2 Benchmark Suites

| Suite | File | What It Tests |
|-------|------|---------------|
| Retrieval | `benches/retrieval_bench.rs` | BM25, hybrid search, scaling |
| Fusion | `benches/fusion_bench.rs` | RRF, weighted sum, multi-method |
| Embedding | `benches/embedding_bench.rs` | Dense/sparse, batch, cache |
| RAPTOR | `benches/raptor_bench.rs` | Tree operations, traversal |
| Ingestion | `benches/ingestion_bench.rs` | Chunking, cleaning, parallel |
| ThinkTools | `benches/thinktool_bench.rs` | Protocol execution, profiles |

### 6.3 Reading Benchmark Results

```
retrieval_bench/sparse_search/query_0
                        time:   [2.3821 ms 2.4234 ms 2.4697 ms]
                        change: [-1.8234% +0.2543% +2.3109%] (p = 0.78 > 0.05)
                        No change in performance detected.
```

- **time**: [lower bound, estimate, upper bound] at 95% confidence
- **change**: Comparison to previous run
- **p-value**: < 0.05 indicates statistically significant change

### 6.4 HTML Reports

After running benchmarks, view detailed reports:

```bash
open target/criterion/report/index.html
```

Reports include:
- Performance trends over time
- Violin plots showing distribution
- Regression analysis
- Historical comparisons

### 6.5 Profiling Tools

#### Flamegraphs (Recommended)

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin rk-core -- query "test query"
```

#### Perf (Linux)

```bash
# Record performance data
perf record -g cargo run --release -- query "test"

# Generate report
perf report
```

#### Instruments (macOS)

```bash
# Time Profiler
xcrun xctrace record --template 'Time Profiler' --launch rk-core query "test"
```

### 6.6 Variance Benchmarking

Test LLM output consistency with structured prompting:

```bash
cd benchmarks
python variance_benchmark.py --runs 20 --model claude-sonnet-4-20250514

# Quick test
python variance_benchmark.py --quick

# Output to specific directory
python variance_benchmark.py -o results/
```

---

## 7. Production Deployment

### 7.1 Recommended Hardware

#### Minimum (Development/Testing)

- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Corpus: Up to 10,000 documents

#### Standard (Production)

- CPU: 8-16 cores
- RAM: 32GB
- Storage: 200GB NVMe
- Corpus: Up to 100,000 documents

#### Enterprise (Large Scale)

- CPU: 32+ cores
- RAM: 128GB+
- Storage: 1TB+ NVMe
- GPU: NVIDIA A100 (for local embeddings)
- Corpus: 1M+ documents

### 7.2 Qdrant Cluster Setup

For high availability and scale:

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant-1:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data_1:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    ports:
      - "6333:6333"
      - "6334:6334"

  qdrant-2:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data_2:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335

volumes:
  qdrant_data_1:
  qdrant_data_2:
```

### 7.3 Connection Pooling

ReasonKit uses HTTP connection pooling for optimal performance:

```rust
// Already configured in llm.rs
reqwest::Client::builder()
    .timeout(Duration::from_secs(120))
    .pool_max_idle_per_host(10)
    .pool_idle_timeout(Duration::from_secs(90))
    .tcp_keepalive(Duration::from_secs(60))
    .build()
```

**Impact**: Eliminates TLS handshake overhead (100-500ms per call).

### 7.4 Monitoring

#### Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Query latency p50 | < 100ms | > 500ms |
| Query latency p99 | < 500ms | > 2s |
| Ingestion rate | > 10 docs/sec | < 1 doc/sec |
| Memory usage | < 80% | > 90% |
| Error rate | < 0.1% | > 1% |
| Cache hit rate | > 80% | < 50% |

#### Prometheus Metrics

```bash
# Enable metrics endpoint
rk-core serve --metrics-port 9090
```

#### Grafana Dashboard

Import the provided dashboard:
```
grafana/dashboards/reasonkit-overview.json
```

---

## 8. Troubleshooting

### 8.1 Common Performance Issues

#### Issue: Slow Query Performance

**Symptoms**: Queries take > 1s (excluding LLM time)

**Diagnosis**:
```bash
# Check collection stats
curl localhost:6333/collections/reasonkit_docs | jq

# Check index status
rk-core stats --verbose
```

**Solutions**:
1. Ensure HNSW index is built: `rk-core index rebuild`
2. Enable quantization for memory pressure
3. Reduce `top_k` if returning too many results
4. Check BM25 index: `rk-core index check-bm25`

#### Issue: High Memory Usage

**Symptoms**: OOM errors or swap thrashing

**Diagnosis**:
```bash
ps aux | grep rk-core
cat /proc/$(pgrep rk-core)/status | grep VmRSS
```

**Solutions**:
1. Enable quantization (`quantization = true`)
2. Reduce embedding dimensions
3. Increase chunk size
4. Use external Qdrant instead of embedded

#### Issue: Slow Ingestion

**Symptoms**: Documents take > 10s each to ingest

**Diagnosis**:
```bash
# Check embedding latency
time rk-core embed "test text"

# Check chunking speed
time rk-core chunk /path/to/doc.md
```

**Solutions**:
1. Increase `batch_size` for API embeddings
2. Use local embeddings with GPU
3. Increase parallel threads
4. Check network latency to embedding API

#### Issue: Inconsistent LLM Outputs

**Symptoms**: Same query produces different results

**Diagnosis**:
```bash
# Run variance benchmark
python benchmarks/variance_benchmark.py --quick
```

**Solutions**:
1. Use structured prompting (ThinkTools profiles)
2. Set `temperature = 0` for deterministic output
3. Enable self-consistency voting
4. Use the `--paranoid` profile for critical queries

### 8.2 Performance Regression Checklist

Before every release:

- [ ] All benchmarks compile without errors
- [ ] No regressions > 5% vs baseline
- [ ] Core loops remain < 5ms
- [ ] Scalability confirmed (linear growth)
- [ ] No memory leaks in batch operations
- [ ] Cache hit rates stable

```bash
# Full quality check
just qa

# Benchmark comparison
cargo bench -- --baseline main
./scripts/run_benchmarks.sh --compare
```

### 8.3 Debug Logging

Enable detailed performance logging:

```bash
RUST_LOG=reasonkit=debug rk-core query "test"

# Even more detail
RUST_LOG=reasonkit=trace rk-core query "test"
```

### 8.4 Getting Help

1. Check the [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues)
2. Review [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
3. Join the [Discord community](https://discord.gg/reasonkit)
4. Email: support@reasonkit.sh

---

## Appendix A: Benchmark File Reference

| File | Purpose | Key Tests |
|------|---------|-----------|
| `benches/retrieval_bench.rs` | Search performance | BM25, hybrid, scaling, concurrent |
| `benches/fusion_bench.rs` | Result fusion | RRF, weighted sum, multi-method |
| `benches/embedding_bench.rs` | Embedding ops | Dense, sparse, batch, cache |
| `benches/raptor_bench.rs` | RAPTOR trees | Create, traverse, lookup |
| `benches/ingestion_bench.rs` | Document processing | Chunk, clean, parallel |
| `benches/thinktool_bench.rs` | ThinkTools protocols | Execute, profile, concurrent |
| `benchmarks/variance_benchmark.py` | LLM consistency | TARa, structured vs raw |

## Appendix B: Performance Optimization Checklist

### Before Deployment

- [ ] Configured appropriate embedding model
- [ ] Set optimal chunk size for content type
- [ ] Enabled quantization if memory-constrained
- [ ] Configured connection pooling
- [ ] Set appropriate HNSW parameters
- [ ] Tested with representative workload

### Ongoing Monitoring

- [ ] Query latency within targets
- [ ] Memory usage stable
- [ ] Cache hit rates acceptable
- [ ] Error rates minimal
- [ ] Ingestion throughput sufficient

### Periodic Maintenance

- [ ] Run full benchmark suite weekly
- [ ] Compare against baseline monthly
- [ ] Review and compact indexes quarterly
- [ ] Update embedding models as needed

---

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Qdrant Performance Tuning](https://qdrant.tech/documentation/guides/performance/)
- [ReasonKit Architecture](../ARCHITECTURE.md)
- [ThinkTools Guide](./THINKTOOLS_GUIDE.md)
- [Benchmark Summary](../BENCHMARK_SUMMARY.md)

---

*"All core loops < 5ms" - This is our contract with users.*

**ReasonKit Performance Engineering**
Version 1.0.0 | https://reasonkit.sh
