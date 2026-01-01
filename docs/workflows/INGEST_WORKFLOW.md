# ReasonKit Ingest-Index-Retrieve Workflow

**Version:** 1.0.0
**Last Updated:** 2026-01-01
**Status:** VERIFIED

## Overview

This document traces the complete Ingest -> Index -> Retrieve workflow through the ReasonKit codebase, identifying entry points, data transformations, and integration touchpoints between `reasonkit-core` and `reasonkit-mem`.

---

## Architecture Diagram

```
                              INGEST-INDEX-RETRIEVE WORKFLOW
                              ==============================

    +-------------------+      +-------------------+      +-------------------+
    |   DOCUMENT        |      |    CHUNKING       |      |   EMBEDDING       |
    |   INGESTION       | ---> |    LOGIC          | ---> |   GENERATION      |
    +-------------------+      +-------------------+      +-------------------+
           |                          |                          |
           v                          v                          v
    +-------------------+      +-------------------+      +-------------------+
    | DocumentIngester  |      | chunk_document()  |      | EmbeddingPipeline |
    | (reasonkit-core)  |      | (reasonkit-core)  |      | (reasonkit-mem)   |
    +-------------------+      +-------------------+      +-------------------+
                                      |                          |
                                      v                          v
                              +-------------------+      +-------------------+
                              |   VECTOR STORAGE  |      |   BM25 INDEXING   |
                              |   (Qdrant/Memory) | <--- |   (Tantivy)       |
                              +-------------------+      +-------------------+
                                      |                          |
                                      +------------+-------------+
                                                   |
                                                   v
                              +-------------------------------------------+
                              |           HYBRID RETRIEVAL                 |
                              |  (Dense + Sparse + RRF Fusion + Rerank)   |
                              +-------------------------------------------+
                                                   |
                                                   v
                              +-------------------------------------------+
                              |              RAG ENGINE                    |
                              |      (Context Assembly + LLM Query)        |
                              +-------------------------------------------+
```

---

## 1. Document Ingestion Entry Point

**Location:** `/reasonkit-core/src/ingestion/mod.rs`

### Entry Point: `DocumentIngester`

```rust
pub struct DocumentIngester {
    pdf_ingester: pdf::PdfIngester,
}

impl DocumentIngester {
    /// Ingest a document from a file path, auto-detecting format
    pub fn ingest(&self, path: &Path) -> Result<Document>
}
```

### Supported Formats

| Extension | Handler Method | Document Type |
|-----------|----------------|---------------|
| `.pdf` | `pdf::PdfIngester::ingest()` | Paper |
| `.md`, `.markdown` | `ingest_markdown()` | Documentation |
| `.html`, `.htm` | `ingest_html()` | Documentation |
| `.json` | `ingest_json()` | Note / Deserialized Document |
| `.jsonl` | `ingest_jsonl()` | Documentation |
| `.txt` | `ingest_text()` | Note |

### Document Data Flow

```
File (Path)
    |
    v
DocumentIngester::ingest()
    |
    v
Format Detection (extension)
    |
    +---> PDF: lopdf parsing + text extraction
    +---> Markdown: pulldown-cmark parsing
    +---> HTML: scraper DOM parsing
    +---> JSON/JSONL: serde deserialization
    +---> Text: direct read
    |
    v
Document {
    id: Uuid,
    doc_type: DocumentType,
    source: Source,
    content: DocumentContent { raw, format, language, word_count, char_count },
    metadata: Metadata { title, authors, abstract, tags, ... },
    processing: ProcessingStatus { pending },
    chunks: Vec<Chunk> (empty at ingestion),
    created_at, updated_at
}
```

---

## 2. Chunking Logic

**Location:** `/reasonkit-core/src/processing/chunking.rs`

### Entry Point: `chunk_document()`

```rust
pub fn chunk_document(
    document: &Document,
    config: &ChunkingConfig,
) -> Result<Vec<Chunk>, ChunkingError>
```

### Chunking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `FixedSize` | Fixed token count per chunk | Simple documents |
| `Semantic` | Split on paragraph/section boundaries | Prose documents |
| `Recursive` | Try different delimiters in order | General purpose (default) |
| `DocumentAware` | Document-type specific logic | Mixed document types |

### Configuration

```rust
pub struct ChunkingConfig {
    pub chunk_size: usize,        // Default: 512 tokens
    pub chunk_overlap: usize,     // Default: 50 tokens
    pub min_chunk_size: usize,    // Default: 100 tokens
    pub strategy: ChunkingStrategy,
    pub respect_sentences: bool,
}
```

### Chunk Output Structure

```rust
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub token_count: Option<usize>,
    pub section: Option<String>,
    pub page: Option<usize>,
    pub embedding_ids: EmbeddingIds,
}
```

### Document-Aware Chunking Logic

```
Document Type       Strategy Applied
--------------      -----------------
Code                chunk_code_aware() - splits on function/class boundaries
Documentation       chunk_markdown_aware() - splits on headers (#, ##, ###)
Paper               chunk_markdown_aware() fallback to chunk_semantic()
Note                chunk_recursive() - tries delimiters in order
Transcript          chunk_recursive()
```

---

## 3. Embedding Generation

**Location:** `/reasonkit-mem/src/embedding/mod.rs`

### Entry Point: `EmbeddingPipeline`

```rust
pub struct EmbeddingPipeline {
    provider: Arc<dyn EmbeddingProvider>,
    batch_size: usize,
}

impl EmbeddingPipeline {
    pub async fn embed_chunks(&self, chunks: &[Chunk]) -> Result<Vec<EmbeddingResult>>
    pub async fn embed_text(&self, text: &str) -> Result<EmbeddingVector>
}
```

### Embedding Providers

| Provider | Configuration | Dimension |
|----------|--------------|-----------|
| OpenAI (`text-embedding-3-small`) | Default | 1536 |
| BGE-M3 (local ONNX) | `feature = "local-embeddings"` | 1024 |
| E5-small (local ONNX) | `feature = "local-embeddings"` | 384 |

### Embedding Flow

```
Chunks (Vec<Chunk>)
    |
    v
EmbeddingPipeline::embed_chunks()
    |
    v
Batch processing (configurable batch_size)
    |
    v
Cache check (SHA256 hash of model + text)
    |
    +---> Cache hit: return cached embedding
    +---> Cache miss: API/local model call
    |
    v
Normalize embeddings (if enabled)
    |
    v
Vec<EmbeddingResult> {
    dense: Option<Vec<f32>>,
    sparse: Option<HashMap<u32, f32>>,
    token_count: usize
}
```

---

## 4. Vector Storage

**Location:** `/reasonkit-mem/src/storage/mod.rs`

### Entry Point: `Storage`

```rust
impl Storage {
    pub fn in_memory() -> Self
    pub async fn file(path: PathBuf) -> Result<Self>

    pub async fn store_document(&self, doc: &Document, ctx: &AccessContext) -> Result<()>
    pub async fn store_embeddings(&self, chunk_id: &Uuid, embedding: &[f32], ctx: &AccessContext) -> Result<()>
    pub async fn search_by_vector(&self, embedding: &[f32], top_k: usize, ctx: &AccessContext) -> Result<Vec<(Uuid, f32)>>
}
```

### Dual-Layer Architecture

```
                  +-------------------+
                  |  DualLayerMemory  |
                  +-------------------+
                          |
                    +-----+-----+
                    |           |
               +-------+   +-------+
               |  Hot  |   | Cold  |
               +-------+   +-------+
                    |           |
                    +-----+-----+
                          |
                     +--------+
                     |  WAL   |
                     +--------+
```

| Layer | Purpose | Characteristics |
|-------|---------|-----------------|
| Hot | Recent/active memories | Fast access, in-memory |
| Cold | Historical/archived | Persistent, optimized storage |
| WAL | Write-ahead log | Durability, crash recovery |

### Backend Options

- **Qdrant** (embedded or cluster mode): Vector similarity search
- **In-Memory**: Development/testing fallback with cosine similarity

---

## 5. Index Building (BM25)

**Location:** `/reasonkit-mem/src/indexing/mod.rs`

### Entry Point: `IndexManager` / `BM25Index`

```rust
impl IndexManager {
    pub fn in_memory() -> Result<Self>
    pub fn open(base_path: PathBuf) -> Result<Self>

    pub fn index_document(&self, doc: &Document) -> Result<usize>
    pub fn search_bm25(&self, query: &str, top_k: usize) -> Result<Vec<BM25Result>>
}
```

### Tantivy Schema

```rust
Schema {
    doc_id: TEXT | STORED,
    chunk_id: TEXT | STORED,
    text: TEXT | STORED,
    section: TEXT | STORED
}
```

### BM25 Index Flow

```
Document + Chunks
    |
    v
BM25Index::index_document()
    |
    v
For each chunk:
    - Create TantivyDocument
    - Add doc_id, chunk_id, text, section
    - writer.add_document()
    |
    v
writer.commit()
    |
    v
Index ready for search
```

---

## 6. Retrieval Query

**Location:** `/reasonkit-mem/src/retrieval/mod.rs`

### Entry Point: `HybridRetriever`

```rust
impl HybridRetriever {
    pub fn in_memory() -> Result<Self>
    pub fn new(storage: Storage, index: IndexManager) -> Self

    pub async fn add_document(&self, doc: &Document) -> Result<()>
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>>
    pub async fn search_sparse(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>>
    pub async fn search_dense(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>>
    pub async fn search_hybrid(&self, query: &str, query_embedding: Option<&[f32]>, config: &RetrievalConfig) -> Result<Vec<HybridResult>>
}
```

### Retrieval Modes

| Mode | Method | Backend Used |
|------|--------|--------------|
| Sparse | `search_sparse()` | BM25 (Tantivy) only |
| Dense | `search_dense()` | Vector (Qdrant) only |
| Hybrid | `search_hybrid()` | Both + RRF Fusion |

### Hybrid Search Flow

```
Query (string)
    |
    v
+---------------------------------------+
|           search_hybrid()             |
+---------------------------------------+
    |
    +---> Sparse: BM25Index::search()
    |         |
    |         v
    |     Vec<BM25Result>
    |
    +---> Dense: Storage::search_by_vector()
    |         |
    |         v
    |     Vec<(Uuid, f32)>
    |
    v
+---------------------------------------+
|           FusionEngine::fuse()        |
|      (Reciprocal Rank Fusion - RRF)   |
+---------------------------------------+
    |
    v
Vec<HybridResult> {
    doc_id: Uuid,
    chunk_id: Uuid,
    text: String,
    score: f32,  // fused score
    dense_score: Option<f32>,
    sparse_score: Option<f32>,
    match_source: MatchSource  // Dense | Sparse | Hybrid
}
```

---

## 7. Result Ranking (Fusion + Reranking)

### Fusion Strategies

**Location:** `/reasonkit-mem/src/retrieval/fusion.rs`

| Strategy | Formula | Default |
|----------|---------|---------|
| RRF (Reciprocal Rank Fusion) | `1 / (k + rank + 1)` | Yes (k=60) |
| Weighted Sum | `dense_weight * dense_score + (1-dense_weight) * sparse_score` | No |
| RBF (Rank-Biased Fusion) | Uses decay parameter `rho` | No |

### Reranking (Optional)

**Location:** `/reasonkit-mem/src/retrieval/rerank.rs`

```rust
pub struct Reranker {
    config: RerankerConfig,
    backend: Arc<dyn CrossEncoderBackend>,
}

impl Reranker {
    pub async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        top_k: usize
    ) -> Result<Vec<RerankedResult>>
}
```

Cross-encoder models score (query, document) pairs together for higher precision.

---

## 8. RAG Engine Integration

**Location:** `/reasonkit-core/src/rag/mod.rs` (feature-gated: `memory`)

### Entry Point: `RagEngine`

```rust
impl RagEngine {
    pub fn in_memory() -> Result<Self>
    pub async fn persistent(base_path: PathBuf) -> Result<Self>

    pub async fn add_document(&self, doc: &Document) -> Result<()>
    pub async fn query(&self, query: &str) -> Result<RagResponse>
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>>
}
```

### RAG Query Flow

```
Query (string)
    |
    v
RagEngine::query()
    |
    v
HybridRetriever::search_sparse() or search_hybrid()
    |
    v
Filter by min_score
    |
    v
Build context from chunks (max_context_tokens)
    |
    v
LLM generation (if client configured)
    |
    v
RagResponse {
    answer: String,
    sources: Vec<RagSource>,
    retrieval_stats: RagRetrievalStats,
    tokens_used: Option<u32>,
    query: String
}
```

---

## Full Workflow Integration

### Complete Data Pipeline

```
1. INGEST
   File Path --> DocumentIngester::ingest() --> Document

2. CHUNK
   Document --> chunk_document() --> Document.chunks = Vec<Chunk>

3. EMBED
   Chunks --> EmbeddingPipeline::embed_chunks() --> Vec<EmbeddingResult>

4. STORE
   Document --> Storage::store_document() --> Qdrant/Memory
   Embeddings --> Storage::store_embeddings() --> Vector Index

5. INDEX
   Document --> IndexManager::index_document() --> Tantivy BM25

6. RETRIEVE
   Query --> HybridRetriever::search_hybrid() --> Vec<HybridResult>

7. RANK
   Results --> FusionEngine::fuse() --> Fused Results
   (Optional) --> Reranker::rerank() --> Reranked Results

8. RAG (Optional)
   Query + Results --> RagEngine::query() --> RagResponse
```

---

## Identified Gaps and Issues

### GAP-001: Automatic Chunking on Ingestion

**Status:** MANUAL
**Description:** `DocumentIngester` returns a `Document` with empty `chunks`. Chunking must be explicitly called via `chunk_document()`.
**Impact:** Users must manually orchestrate ingestion -> chunking flow.
**Recommendation:** Consider an `IngestPipeline` that combines ingestion + chunking.

### GAP-002: Missing Embedding Integration in Ingestion

**Status:** DISCONNECTED
**Description:** Ingestion module (`reasonkit-core`) does not invoke embedding generation. Embeddings are generated separately via `reasonkit-mem`.
**Impact:** Requires explicit orchestration by caller.
**Recommendation:** Document the expected usage pattern clearly, or provide a unified `IngestPipeline`.

### GAP-003: No Automatic Index Update

**Status:** MANUAL
**Description:** Adding a document to storage does not automatically update the BM25 index unless using `HybridRetriever::add_document()`.
**Impact:** Direct storage usage skips indexing.
**Recommendation:** Always use `HybridRetriever` or `KnowledgeBase` for document operations.

### GAP-004: Type Conversion Between Crates

**Status:** IMPLEMENTED BUT VERBOSE
**Description:** `reasonkit-core::Document` must be converted to `reasonkit_mem::Document` using `From` trait.
**Impact:** Boilerplate code required.
**Recommendation:** Consider a shared types crate or re-export strategy.

---

## Test Coverage Analysis

### Existing Tests

| Module | Test File | Coverage |
|--------|-----------|----------|
| Ingestion | `ingestion/mod.rs` (unit tests) | Markdown, Text |
| Chunking | `processing/chunking.rs` (unit tests) | All strategies |
| Embedding | `embedding/mod.rs` (unit tests) | Normalization, cache |
| Indexing | `indexing/mod.rs` (unit tests) | BM25 search |
| Retrieval | `retrieval/mod.rs` (unit tests) | Hybrid, KnowledgeBase |
| Storage | `tests/storage_integration_tests.rs` | Full workflow |
| RAG | `rag/mod.rs` (unit tests) | Engine, retrieve |

### Missing E2E Test

A full end-to-end test covering:
1. File ingestion
2. Chunking
3. Embedding (mock or real)
4. Storage + Indexing
5. Hybrid retrieval
6. Result ranking

**See:** `tests/ingest_workflow_e2e_test.rs` (created with this document)

---

## Usage Examples

### Basic Workflow (Manual Orchestration)

```rust
use reasonkit_core::{
    ingestion::DocumentIngester,
    processing::chunking::{chunk_document, ChunkingConfig},
};
use reasonkit_mem::{
    retrieval::{HybridRetriever, KnowledgeBase},
    embedding::{EmbeddingPipeline, OpenAIEmbedding},
};
use std::sync::Arc;

// 1. Ingest
let ingester = DocumentIngester::new();
let mut doc = ingester.ingest(Path::new("paper.pdf"))?;

// 2. Chunk
let config = ChunkingConfig::default();
doc.chunks = chunk_document(&doc, &config)?;

// 3. Create retriever with embedding pipeline
let embedding = Arc::new(EmbeddingPipeline::new(
    Arc::new(OpenAIEmbedding::openai()?)
));
let retriever = HybridRetriever::in_memory()?
    .with_embedding_pipeline(embedding);

// 4. Add document (stores, indexes, embeds)
let mem_doc: reasonkit_mem::Document = doc.into();
retriever.add_document(&mem_doc).await?;

// 5. Search
let results = retriever.search("quantum computing", 10).await?;
```

### Using KnowledgeBase (Recommended)

```rust
use reasonkit_mem::retrieval::KnowledgeBase;

let kb = KnowledgeBase::in_memory()?
    .with_embedding_pipeline(embedding);

kb.add(&doc).await?;
let results = kb.query("What is RAPTOR?", 5).await?;
```

### Using RAG Engine

```rust
use reasonkit::rag::{RagEngine, RagConfig};

let engine = RagEngine::in_memory()?
    .with_llm(llm_client)
    .with_config(RagConfig::thorough());

engine.add_document(&doc).await?;
let response = engine.query("Explain chain-of-thought reasoning").await?;

println!("Answer: {}", response.answer);
for source in response.sources {
    println!("- [{}] {}", source.score, source.text);
}
```

---

## Appendix: Key File Locations

| Component | File Path |
|-----------|-----------|
| Document Ingestion | `reasonkit-core/src/ingestion/mod.rs` |
| PDF Ingestion | `reasonkit-core/src/ingestion/pdf.rs` |
| Chunking | `reasonkit-core/src/processing/chunking.rs` |
| Text Processing | `reasonkit-core/src/processing/mod.rs` |
| Embedding | `reasonkit-mem/src/embedding/mod.rs` |
| Embedding Cache | `reasonkit-mem/src/embedding/cache.rs` |
| Storage | `reasonkit-mem/src/storage/mod.rs` |
| BM25 Indexing | `reasonkit-mem/src/indexing/mod.rs` |
| Hybrid Retrieval | `reasonkit-mem/src/retrieval/mod.rs` |
| Fusion | `reasonkit-mem/src/retrieval/fusion.rs` |
| Reranking | `reasonkit-mem/src/retrieval/rerank.rs` |
| RAG Engine | `reasonkit-core/src/rag/mod.rs` |
| Core Types | `reasonkit-core/src/lib.rs` |
| Memory Types | `reasonkit-mem/src/types.rs` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-01 | Initial workflow documentation |
