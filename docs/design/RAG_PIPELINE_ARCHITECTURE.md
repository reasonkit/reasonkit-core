# RAG PIPELINE ARCHITECTURE DESIGN SPECIFICATION
> ReasonKit Core - 5-Layer Retrieval-Augmented Generation System
> Version: 1.0.0 | Status: Design Phase
> Author: ReasonKit Team

---

## 1. EXECUTIVE SUMMARY

This document specifies the complete RAG (Retrieval-Augmented Generation) pipeline
architecture for ReasonKit Core. The design implements a 5-layer architecture with
RAPTOR hierarchical retrieval, hybrid search (BM25 + dense vectors), and production-grade
error handling.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REASONKIT RAG PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INGEST    │ -> │  PROCESS    │ -> │   EMBED     │ -> │   INDEX     │  │
│  │  (Layer 1)  │    │  (Layer 2)  │    │  (Layer 3)  │    │  (Layer 4)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STORAGE LAYER                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │    Qdrant    │  │   Tantivy    │  │ RAPTOR Tree  │               │   │
│  │  │  (Vectors)   │  │   (BM25)     │  │ (Summaries)  │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─────────────────┐                                │
│                          │    RETRIEVE     │                                │
│                          │    (Layer 5)    │                                │
│                          └─────────────────┘                                │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────────┐          ┌─────────────┐     │
│  │   Vector    │          │  Hybrid Fusion  │          │   Rerank    │     │
│  │   Search    │    ->    │    (RRF/α)      │    ->    │  (Optional) │     │
│  │  + BM25     │          │                 │          │             │     │
│  └─────────────┘          └─────────────────┘          └─────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 Layer Summary

| Layer | Name | Input | Output | Key Components |
|-------|------|-------|--------|----------------|
| **1** | Ingestion | Raw files | Parsed documents | PDF, MD, HTML parsers |
| **2** | Processing | Documents | Chunks + metadata | Chunking, cleaning, extraction |
| **3** | Embedding | Chunks | Vectors | OpenAI, local ONNX models |
| **4** | Indexing | Vectors + text | Searchable indexes | HNSW, BM25, RAPTOR tree |
| **5** | Retrieval | Query | Ranked results | Hybrid search, reranking |

### 2.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Modularity** | Each layer is independently testable and replaceable |
| **Async-first** | All I/O operations use async/await |
| **Streaming** | Large documents processed as streams, not loaded into memory |
| **Fail-fast** | Errors propagate immediately with full context |
| **Observability** | Tracing spans for every operation |

---

## 3. LAYER 1: INGESTION

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: INGESTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT SOURCES                    PARSERS                    OUTPUT         │
│  ─────────────                    ───────                    ──────         │
│                                                                             │
│  ┌──────────┐                  ┌──────────────┐                             │
│  │   PDF    │ ───────────────> │  PdfParser   │ ─┐                          │
│  └──────────┘                  │  (lopdf)     │  │                          │
│                                └──────────────┘  │                          │
│  ┌──────────┐                  ┌──────────────┐  │     ┌──────────────────┐ │
│  │ Markdown │ ───────────────> │  MdParser    │  ├───> │    Document      │ │
│  └──────────┘                  │ (pulldown)   │  │     │                  │ │
│                                └──────────────┘  │     │ - id: UUID       │ │
│  ┌──────────┐                  ┌──────────────┐  │     │ - content: String│ │
│  │   HTML   │ ───────────────> │  HtmlParser  │  │     │ - metadata: Map  │ │
│  └──────────┘                  │  (scraper)   │  │     │ - source: String │ │
│                                └──────────────┘  │     │ - doc_type: Enum │ │
│  ┌──────────┐                  ┌──────────────┐  │     │ - created: Time  │ │
│  │   JSON   │ ───────────────> │  JsonParser  │ ─┘     └──────────────────┘ │
│  └──────────┘                  │  (serde)     │                             │
│                                └──────────────┘                             │
│  ┌──────────┐                  ┌──────────────┐                             │
│  │  GitHub  │ ───────────────> │  GithubFetch │ ─────> (same as above)      │
│  └──────────┘                  │  (octocrab)  │                             │
│                                └──────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Rust Trait Definitions

```rust
//! Layer 1: Ingestion traits and types

use async_trait::async_trait;
use std::path::Path;

/// Core document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: Uuid,

    /// Raw text content
    pub content: String,

    /// Document metadata
    pub metadata: DocumentMetadata,

    /// Original source (path, URL, etc.)
    pub source: String,

    /// Document type
    pub doc_type: DocumentType,

    /// Ingestion timestamp
    pub created_at: DateTime<Utc>,

    /// Content hash for deduplication
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title (extracted or inferred)
    pub title: Option<String>,

    /// Author(s)
    pub authors: Vec<String>,

    /// Publication date
    pub date: Option<NaiveDate>,

    /// Tags/categories
    pub tags: Vec<String>,

    /// Custom key-value pairs
    pub custom: HashMap<String, serde_json::Value>,

    /// File size in bytes
    pub file_size: Option<u64>,

    /// Original file format
    pub format: String,

    /// Language (ISO 639-1)
    pub language: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentType {
    Paper,       // Academic papers
    Documentation, // Technical docs
    Code,        // Source code
    Note,        // Personal notes
    Web,         // Web content
    Unknown,
}

/// Parser trait for document ingestion
#[async_trait]
pub trait DocumentParser: Send + Sync {
    /// Parse a file into a document
    async fn parse_file(&self, path: &Path) -> Result<Document, IngestionError>;

    /// Parse raw bytes with format hint
    async fn parse_bytes(&self, bytes: &[u8], format: &str) -> Result<Document, IngestionError>;

    /// Check if this parser can handle the given format
    fn can_parse(&self, format: &str) -> bool;

    /// Supported file extensions
    fn supported_extensions(&self) -> &[&str];
}

/// Ingestion pipeline orchestrator
#[async_trait]
pub trait Ingestor: Send + Sync {
    /// Ingest a single file
    async fn ingest_file(&self, path: &Path, options: IngestOptions) -> Result<Document, IngestionError>;

    /// Ingest a directory (optionally recursive)
    async fn ingest_dir(&self, path: &Path, options: IngestOptions) -> Result<Vec<Document>, IngestionError>;

    /// Ingest from URL
    async fn ingest_url(&self, url: &str, options: IngestOptions) -> Result<Document, IngestionError>;

    /// Ingest from GitHub repository
    async fn ingest_github(&self, repo: &str, options: IngestOptions) -> Result<Vec<Document>, IngestionError>;
}

#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub doc_type: Option<DocumentType>,
    pub tags: Vec<String>,
    pub recursive: bool,
    pub include_glob: Option<String>,
    pub exclude_glob: Option<String>,
    pub parallel: usize,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            doc_type: None,
            tags: vec![],
            recursive: true,
            include_glob: None,
            exclude_glob: None,
            parallel: 4,
        }
    }
}
```

### 3.3 Implementation Requirements

| Requirement | Details |
|-------------|---------|
| PDF Parsing | Use `lopdf` for text extraction; handle OCR fallback |
| Markdown | Use `pulldown-cmark` with GFM extensions |
| HTML | Use `scraper` for DOM parsing; extract main content |
| Deduplication | SHA-256 content hash to prevent duplicates |
| Streaming | Process files > 10MB as streams |
| Error Recovery | Continue on individual file errors |

---

## 4. LAYER 2: PROCESSING

### 4.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 2: PROCESSING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐          PROCESSING PIPELINE         ┌──────────────────┐ │
│  │   Document   │                                      │     Chunks       │ │
│  │              │  ┌─────────────────────────────────┐ │                  │ │
│  │ - content    │  │                                 │ │ Vec<Chunk>       │ │
│  │ - metadata   │  │  ┌─────────┐   ┌───────────┐   │ │                  │ │
│  │ - source     │ ─>  │ CLEAN   │ ─>│  CHUNK    │   │ │ - id: UUID       │ │
│  │              │  │  │         │   │           │   │ │ - content: String│ │
│  └──────────────┘  │  │ - Strip │   │ - Size    │   │ │ - doc_id: UUID   │ │
│                    │  │   noise │   │ - Overlap │   │ │ - index: usize   │ │
│                    │  │ - Norm  │   │ - Semantic│   │ │ - metadata: Map  │ │
│                    │  │   space │   │   aware   │ ──│ │ - start_char: u  │ │
│                    │  └─────────┘   └───────────┘   │ │ - end_char: usize│ │
│                    │                                 │ │                  │ │
│                    │  ┌─────────────────────────┐    │ └──────────────────┘ │
│                    │  │     METADATA EXTRACT    │    │                      │
│                    │  │                         │    │                      │
│                    │  │ - Title extraction      │ ───┘                      │
│                    │  │ - Author detection      │                           │
│                    │  │ - Date parsing          │                           │
│                    │  │ - Section headers       │                           │
│                    │  │ - Citation extraction   │                           │
│                    │  └─────────────────────────┘                           │
│                                                                             │
│  CHUNKING STRATEGIES:                                                       │
│  ─────────────────────                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  1. FIXED SIZE (default)                                             │   │
│  │     └── 512 tokens, 50 token overlap                                 │   │
│  │                                                                      │   │
│  │  2. SEMANTIC CHUNKING                                                │   │
│  │     └── Split on paragraph/section boundaries                        │   │
│  │     └── Preserve semantic units                                      │   │
│  │                                                                      │   │
│  │  3. RECURSIVE CHARACTER SPLITTING                                    │   │
│  │     └── Try: \n\n -> \n -> . -> space                                │   │
│  │     └── Fallback to character split                                  │   │
│  │                                                                      │   │
│  │  4. DOCUMENT-AWARE                                                   │   │
│  │     └── Code: function/class boundaries                              │   │
│  │     └── Markdown: header boundaries                                  │   │
│  │     └── Papers: section boundaries                                   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Rust Trait Definitions

```rust
//! Layer 2: Processing traits and types

/// A chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: Uuid,

    /// Text content
    pub content: String,

    /// Parent document ID
    pub document_id: Uuid,

    /// Index within document
    pub chunk_index: usize,

    /// Total chunks in document
    pub total_chunks: usize,

    /// Character offset start
    pub start_char: usize,

    /// Character offset end
    pub end_char: usize,

    /// Token count (approximate)
    pub token_count: usize,

    /// Inherited + chunk-specific metadata
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Section header (if identifiable)
    pub section: Option<String>,

    /// Subsection
    pub subsection: Option<String>,

    /// Is this a summary chunk (RAPTOR)?
    pub is_summary: bool,

    /// RAPTOR tree level (0 = leaf)
    pub raptor_level: u8,

    /// Previous chunk ID (for context)
    pub prev_chunk_id: Option<Uuid>,

    /// Next chunk ID (for context)
    pub next_chunk_id: Option<Uuid>,
}

/// Configuration for chunking
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Target chunk size in tokens
    pub chunk_size: usize,

    /// Overlap between chunks in tokens
    pub chunk_overlap: usize,

    /// Minimum chunk size (don't create tiny chunks)
    pub min_chunk_size: usize,

    /// Chunking strategy
    pub strategy: ChunkingStrategy,

    /// Preserve sentence boundaries
    pub respect_sentences: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ChunkingStrategy {
    /// Fixed token size
    FixedSize,

    /// Split on semantic boundaries
    Semantic,

    /// Recursive character splitting
    Recursive,

    /// Document-type aware
    DocumentAware,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 100,
            strategy: ChunkingStrategy::Recursive,
            respect_sentences: true,
        }
    }
}

/// Text processor trait
#[async_trait]
pub trait TextProcessor: Send + Sync {
    /// Clean text (normalize whitespace, remove artifacts)
    fn clean(&self, text: &str) -> String;

    /// Chunk a document into smaller pieces
    fn chunk(&self, document: &Document, config: &ChunkingConfig) -> Vec<Chunk>;

    /// Extract metadata from text
    fn extract_metadata(&self, text: &str) -> DocumentMetadata;
}

/// Document processor orchestrator
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process a document into chunks
    async fn process(&self, document: Document, config: ProcessConfig) -> Result<Vec<Chunk>, ProcessingError>;

    /// Process multiple documents in parallel
    async fn process_batch(&self, documents: Vec<Document>, config: ProcessConfig) -> Result<Vec<Chunk>, ProcessingError>;
}

#[derive(Debug, Clone)]
pub struct ProcessConfig {
    pub chunking: ChunkingConfig,
    pub extract_metadata: bool,
    pub clean_text: bool,
    pub parallel: usize,
}
```

### 4.3 Processing Pipeline

```
Document
    │
    ▼
┌───────────────────────────────────┐
│ 1. TEXT CLEANING                  │
│    - Normalize Unicode            │
│    - Remove control characters    │
│    - Collapse whitespace          │
│    - Strip headers/footers (PDF)  │
│    - Remove boilerplate (HTML)    │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│ 2. METADATA EXTRACTION            │
│    - Title (regex + heuristics)   │
│    - Authors (NER or patterns)    │
│    - Dates (dateparser)           │
│    - Section headers              │
│    - References/citations         │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│ 3. CHUNKING                       │
│    - Apply strategy               │
│    - Respect boundaries           │
│    - Add overlap                  │
│    - Link prev/next               │
└───────────────────────────────────┘
    │
    ▼
Vec<Chunk>
```

---

## 5. LAYER 3: EMBEDDING

### 5.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 3: EMBEDDING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   Chunks    │                                                            │
│  │  Vec<Chunk> │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    EMBEDDING ROUTER                                  │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ config.embedding.provider = ?                               │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │              │                    │                    │             │   │
│  │              ▼                    ▼                    ▼             │   │
│  │   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐    │   │
│  │   │     OPENAI       │ │      LOCAL       │ │     COHERE       │    │   │
│  │   │                  │ │                  │ │                  │    │   │
│  │   │ text-embed-3-sm  │ │   BGE-M3 ONNX    │ │  embed-english   │    │   │
│  │   │ text-embed-3-lg  │ │   E5-large       │ │  embed-multi     │    │   │
│  │   │ ada-002          │ │   Qwen3-embed    │ │                  │    │   │
│  │   │                  │ │                  │ │                  │    │   │
│  │   │ Batch: 2048      │ │ Batch: 32        │ │ Batch: 96        │    │   │
│  │   │ Rate limited     │ │ Local inference  │ │ Rate limited     │    │   │
│  │   └──────────────────┘ └──────────────────┘ └──────────────────┘    │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       EMBEDDING OUTPUT                               │   │
│  │                                                                      │   │
│  │   Vec<ChunkEmbedding>                                                │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ chunk_id: UUID                                              │   │   │
│  │   │ embedding: Vec<f32>  // 1536 dimensions (OpenAI)            │   │   │
│  │   │ model: String        // "text-embedding-3-small"            │   │   │
│  │   │ dimensions: usize    // 1536                                │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Rust Trait Definitions

```rust
//! Layer 3: Embedding traits and types

/// Embedding vector for a chunk
#[derive(Debug, Clone)]
pub struct ChunkEmbedding {
    /// Chunk ID this embedding belongs to
    pub chunk_id: Uuid,

    /// The embedding vector
    pub embedding: Vec<f32>,

    /// Model used to generate embedding
    pub model: String,

    /// Vector dimensions
    pub dimensions: usize,

    /// Generation timestamp
    pub created_at: DateTime<Utc>,
}

/// Embedding model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider (openai, local, cohere)
    pub provider: EmbeddingProvider,

    /// Model identifier
    pub model: String,

    /// Output dimensions
    pub dimensions: usize,

    /// Batch size for embedding
    pub batch_size: usize,

    /// API configuration
    #[serde(flatten)]
    pub api: Option<ApiConfig>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum EmbeddingProvider {
    OpenAI,
    Local,
    Cohere,
    Custom,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: usize,
}

/// Embedding service trait
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embed multiple texts in batch
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Embed chunks and return chunk embeddings
    async fn embed_chunks(&self, chunks: &[Chunk]) -> Result<Vec<ChunkEmbedding>, EmbeddingError>;

    /// Get model information
    fn model_info(&self) -> EmbeddingModelInfo;
}

#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    pub name: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub provider: EmbeddingProvider,
}

/// Embedding with retry and rate limiting
pub struct EmbeddingService {
    embedder: Box<dyn Embedder>,
    rate_limiter: RateLimiter,
    retry_config: RetryConfig,
}

impl EmbeddingService {
    pub async fn embed_with_retry(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut attempts = 0;
        loop {
            self.rate_limiter.acquire().await;
            match self.embedder.embed(text).await {
                Ok(embedding) => return Ok(embedding),
                Err(e) if e.is_retryable() && attempts < self.retry_config.max_retries => {
                    attempts += 1;
                    let delay = self.retry_config.backoff(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```

### 5.3 Embedding Models Comparison

| Model | Provider | Dimensions | Max Tokens | Cost/1M | Use Case |
|-------|----------|------------|------------|---------|----------|
| text-embedding-3-small | OpenAI | 1536 | 8191 | $0.02 | General purpose |
| text-embedding-3-large | OpenAI | 3072 | 8191 | $0.13 | High accuracy |
| BGE-M3 | Local | 1024 | 8192 | Free | Self-hosted |
| Qwen3-Embed-8B | Local | 4096 | 32768 | Free | Long context |

---

## 6. LAYER 4: INDEXING

### 6.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 4: INDEXING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Vec<Chunk> + Vec<ChunkEmbedding>                                    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      INDEX MANAGER                                   │   │
│  │                                                                      │   │
│  │   Coordinates all indexing operations                                │   │
│  │   Maintains consistency across index types                           │   │
│  │                                                                      │   │
│  └───────────────────────────┬──────────────────────────────────────────┘   │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                         │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  VECTOR      │    │    BM25      │    │   RAPTOR     │                   │
│  │  INDEX       │    │    INDEX     │    │    TREE      │                   │
│  │              │    │              │    │              │                   │
│  │  ┌────────┐  │    │  ┌────────┐  │    │  ┌────────┐  │                   │
│  │  │ Qdrant │  │    │  │Tantivy │  │    │  │ Custom │  │                   │
│  │  │  HNSW  │  │    │  │  BM25  │  │    │  │  Tree  │  │                   │
│  │  └────────┘  │    │  └────────┘  │    │  └────────┘  │                   │
│  │              │    │              │    │              │                   │
│  │ Dense search │    │ Sparse search│    │ Hierarchical │                   │
│  │ ANN queries  │    │ Keyword match│    │  summaries   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  RAPTOR TREE STRUCTURE:                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│                        ┌─────────────────┐                                  │
│            Level 2     │  Root Summary   │  (1 summary of everything)       │
│                        └────────┬────────┘                                  │
│                                 │                                           │
│                    ┌────────────┼────────────┐                              │
│                    │            │            │                              │
│            ┌───────▼──┐  ┌──────▼───┐  ┌────▼─────┐                         │
│  Level 1   │ Summary  │  │ Summary  │  │ Summary  │  (cluster summaries)    │
│            │ Cluster1 │  │ Cluster2 │  │ Cluster3 │                         │
│            └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
│                 │             │             │                               │
│       ┌─────┬───┴───┬───┐   ┌─┴─┐...     ┌─┴─┐...                           │
│       │     │       │   │   │   │        │   │                              │
│       ▼     ▼       ▼   ▼   ▼   ▼        ▼   ▼                              │
│  Level 0  [C1] [C2] [C3] [C4] [C5]...  [Cn]    (original chunks)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Rust Trait Definitions

```rust
//! Layer 4: Indexing traits and types

/// Vector index trait (Qdrant, HNSW, etc.)
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Insert a batch of embeddings
    async fn upsert(&self, embeddings: &[ChunkEmbedding]) -> Result<(), IndexError>;

    /// Search for similar vectors
    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<VectorSearchResult>, IndexError>;

    /// Delete embeddings by chunk IDs
    async fn delete(&self, chunk_ids: &[Uuid]) -> Result<(), IndexError>;

    /// Get index statistics
    async fn stats(&self) -> Result<IndexStats, IndexError>;
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub chunk_id: Uuid,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

/// BM25 index trait (Tantivy)
#[async_trait]
pub trait BM25Index: Send + Sync {
    /// Index chunks
    async fn index(&self, chunks: &[Chunk]) -> Result<(), IndexError>;

    /// Search with BM25
    async fn search(
        &self,
        query: &str,
        top_k: usize,
        fields: &[&str],
    ) -> Result<Vec<BM25SearchResult>, IndexError>;

    /// Delete by chunk IDs
    async fn delete(&self, chunk_ids: &[Uuid]) -> Result<(), IndexError>;

    /// Optimize index
    async fn optimize(&self) -> Result<(), IndexError>;
}

#[derive(Debug, Clone)]
pub struct BM25SearchResult {
    pub chunk_id: Uuid,
    pub score: f32,
    pub highlights: Vec<String>,
}

/// RAPTOR tree for hierarchical retrieval
#[async_trait]
pub trait RaptorTree: Send + Sync {
    /// Build tree from chunks
    async fn build(&self, chunks: &[Chunk], embeddings: &[ChunkEmbedding]) -> Result<(), IndexError>;

    /// Query the tree (returns chunks from multiple levels)
    async fn query(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<RaptorResult>, IndexError>;

    /// Get tree statistics
    async fn stats(&self) -> Result<RaptorStats, IndexError>;

    /// Rebuild tree (useful after significant updates)
    async fn rebuild(&self) -> Result<(), IndexError>;
}

#[derive(Debug, Clone)]
pub struct RaptorResult {
    pub chunk_id: Uuid,
    pub score: f32,
    pub level: u8,        // 0 = leaf, 1+ = summary
    pub is_summary: bool,
}

#[derive(Debug)]
pub struct RaptorStats {
    pub levels: usize,
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub summary_nodes: usize,
}

/// Index manager orchestrating all indexes
pub struct IndexManager {
    vector_index: Box<dyn VectorIndex>,
    bm25_index: Box<dyn BM25Index>,
    raptor_tree: Option<Box<dyn RaptorTree>>,
}

impl IndexManager {
    /// Index chunks across all index types
    pub async fn index_chunks(
        &self,
        chunks: &[Chunk],
        embeddings: &[ChunkEmbedding],
    ) -> Result<(), IndexError> {
        // Run in parallel
        let (vector_result, bm25_result) = tokio::join!(
            self.vector_index.upsert(embeddings),
            self.bm25_index.index(chunks),
        );

        vector_result?;
        bm25_result?;

        // Build RAPTOR tree if enabled
        if let Some(ref tree) = self.raptor_tree {
            tree.build(chunks, embeddings).await?;
        }

        Ok(())
    }
}
```

### 6.3 RAPTOR Algorithm

```
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
═══════════════════════════════════════════════════════════════════════

INPUT: Vec<Chunk> (leaf nodes)

ALGORITHM:
1. Embed all chunks
2. Cluster chunks using k-means or HDBSCAN
3. For each cluster:
   a. Concatenate chunk contents
   b. Generate summary using LLM
   c. Embed summary
4. Recursively apply steps 2-3 to summaries
5. Stop when root summary reached (or max levels)

RETRIEVAL:
- Query embedding compared against ALL levels
- Results from multiple levels merged
- Higher-level matches provide context
- Lower-level matches provide detail

BENEFITS:
- Captures both fine-grained and high-level semantics
- +20% improvement on QuALITY benchmark (per RAPTOR paper)
- Handles long documents better than flat chunking
```

---

## 7. LAYER 5: RETRIEVAL

### 7.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 5: RETRIEVAL                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Query string                                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    RETRIEVAL PIPELINE                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 1: QUERY PROCESSING                                                   │
│  ─────────────────────────                                                  │
│  ┌─────────────────┐                                                        │
│  │  Query: "..."   │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ - Clean/normalize query                                             │   │
│  │ - Expand query (optional synonyms)                                  │   │
│  │ - Generate query embedding                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: PARALLEL SEARCH                                                    │
│  ───────────────────────                                                    │
│           │                                                                 │
│           ├─────────────────────┬─────────────────────┐                     │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  VECTOR SEARCH  │   │   BM25 SEARCH   │   │  RAPTOR QUERY   │           │
│  │                 │   │                 │   │   (optional)    │           │
│  │  top_k * 2      │   │  top_k * 2      │   │  top_k results  │           │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│           │                     │                     │                     │
│           └─────────────────────┴─────────────────────┘                     │
│                                 │                                           │
│  STEP 3: FUSION                 ▼                                           │
│  ──────────────    ┌─────────────────────────────────────────────────────┐ │
│                    │              HYBRID FUSION                          │ │
│                    │                                                     │ │
│                    │  Method: RRF (Reciprocal Rank Fusion)               │ │
│                    │                                                     │ │
│                    │  RRF_score(d) = Σ 1/(k + rank_i(d))                 │ │
│                    │                i                                    │ │
│                    │                                                     │ │
│                    │  OR                                                 │ │
│                    │                                                     │ │
│                    │  Method: Linear combination                         │ │
│                    │                                                     │ │
│                    │  score(d) = α × vector_score + (1-α) × bm25_score   │ │
│                    │                                                     │ │
│                    └─────────────────────────────────────────────────────┘ │
│                                 │                                           │
│  STEP 4: RERANKING             ▼                                           │
│  ─────────────────  ┌─────────────────────────────────────────────────────┐│
│    (Optional)       │              CROSS-ENCODER RERANKING                ││
│                     │                                                     ││
│                     │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2        ││
│                     │                                                     ││
│                     │  For each candidate:                                ││
│                     │    score = CrossEncoder(query, document)            ││
│                     │                                                     ││
│                     │  Sort by rerank score                               ││
│                     └─────────────────────────────────────────────────────┘│
│                                 │                                           │
│                                 ▼                                           │
│  OUTPUT:          ┌─────────────────────────────────────────────────────┐   │
│                   │  Vec<SearchResult>                                  │   │
│                   │                                                     │   │
│                   │  - chunk_id                                         │   │
│                   │  - content                                          │   │
│                   │  - score (final)                                    │   │
│                   │  - metadata                                         │   │
│                   │  - search_type (vector/bm25/hybrid/raptor)          │   │
│                   └─────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Rust Trait Definitions

```rust
//! Layer 5: Retrieval traits and types

/// Search result from retrieval
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// Chunk ID
    pub chunk_id: Uuid,

    /// Chunk content
    pub content: String,

    /// Final relevance score
    pub score: f32,

    /// Search type that found this result
    pub search_type: SearchType,

    /// Metadata
    pub metadata: serde_json::Value,

    /// Score breakdown (for explain mode)
    pub score_breakdown: Option<ScoreBreakdown>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SearchType {
    Vector,
    BM25,
    Hybrid,
    Raptor,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreBreakdown {
    pub vector_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub raptor_score: Option<f32>,
    pub rerank_score: Option<f32>,
    pub final_score: f32,
}

/// Retrieval configuration
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of results to return
    pub top_k: usize,

    /// Use hybrid search
    pub hybrid: bool,

    /// Hybrid fusion weight (0 = BM25 only, 1 = vector only)
    pub alpha: f32,

    /// Fusion method
    pub fusion_method: FusionMethod,

    /// Use RAPTOR tree
    pub use_raptor: bool,

    /// Apply reranking
    pub rerank: bool,

    /// Reranking model
    pub rerank_model: Option<String>,

    /// Number of candidates before reranking
    pub rerank_candidates: usize,

    /// Metadata filter
    pub filter: Option<MetadataFilter>,

    /// Minimum score threshold
    pub min_score: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion
    RRF { k: usize },
    /// Linear combination
    Linear,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            hybrid: true,
            alpha: 0.5,
            fusion_method: FusionMethod::RRF { k: 60 },
            use_raptor: false,
            rerank: false,
            rerank_model: None,
            rerank_candidates: 50,
            filter: None,
            min_score: 0.0,
        }
    }
}

/// Main retriever trait
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Perform retrieval
    async fn retrieve(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<Vec<SearchResult>, RetrievalError>;

    /// Explain retrieval (for debugging)
    async fn retrieve_with_explanation(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<RetrievalExplanation, RetrievalError>;
}

#[derive(Debug, Serialize)]
pub struct RetrievalExplanation {
    pub query: String,
    pub query_embedding_time_ms: u64,
    pub vector_search_time_ms: u64,
    pub bm25_search_time_ms: u64,
    pub fusion_time_ms: u64,
    pub rerank_time_ms: Option<u64>,
    pub total_time_ms: u64,
    pub candidates_before_fusion: usize,
    pub candidates_after_fusion: usize,
    pub results: Vec<SearchResult>,
}

/// Hybrid retriever implementation
pub struct HybridRetriever {
    vector_index: Arc<dyn VectorIndex>,
    bm25_index: Arc<dyn BM25Index>,
    raptor_tree: Option<Arc<dyn RaptorTree>>,
    embedder: Arc<dyn Embedder>,
    reranker: Option<Arc<dyn Reranker>>,
}

#[async_trait]
impl Retriever for HybridRetriever {
    async fn retrieve(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        // 1. Generate query embedding
        let query_embedding = self.embedder.embed(query).await?;

        // 2. Parallel search
        let candidates = config.rerank_candidates.max(config.top_k * 2);

        let (vector_results, bm25_results, raptor_results) = tokio::join!(
            self.vector_index.search(&query_embedding, candidates, config.filter.as_ref()),
            self.bm25_index.search(query, candidates, &["content"]),
            self.search_raptor(&query_embedding, candidates, config),
        );

        // 3. Fusion
        let fused = self.fuse_results(
            vector_results?,
            bm25_results?,
            raptor_results?,
            config,
        )?;

        // 4. Rerank (optional)
        let results = if config.rerank {
            self.rerank(&fused, query, config.top_k).await?
        } else {
            fused.into_iter().take(config.top_k).collect()
        };

        Ok(results)
    }
}
```

### 7.3 Fusion Algorithm (RRF)

```rust
/// Reciprocal Rank Fusion
pub fn rrf_fusion(
    result_lists: Vec<Vec<(Uuid, f32)>>,
    k: usize,
) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();

    for results in result_lists {
        for (rank, (id, _original_score)) in results.into_iter().enumerate() {
            let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);
            *scores.entry(id).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    fused
}
```

---

## 8. END-TO-END PIPELINE

### 8.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE RAG PIPELINE DATA FLOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INGESTION FLOW (Write Path)                                                │
│  ═══════════════════════════                                                │
│                                                                             │
│  Files/URLs                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │ INGEST  │ -> │ PROCESS │ -> │  EMBED  │ -> │  INDEX  │ -> │ STORAGE │   │
│  │         │    │         │    │         │    │         │    │         │   │
│  │ Parse   │    │ Clean   │    │ Vectors │    │ HNSW    │    │ Qdrant  │   │
│  │ Extract │    │ Chunk   │    │ 1536d   │    │ BM25    │    │ Tantivy │   │
│  │ Validate│    │ Metadata│    │         │    │ RAPTOR  │    │ Files   │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  RETRIEVAL FLOW (Read Path)                                                 │
│  ══════════════════════════                                                 │
│                                                                             │
│      Query                                                                  │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  EMBED   │ ->│  SEARCH  │ ->│  FUSION  │ ->│ RERANK   │ ->│ RESULTS  │ │
│  │  QUERY   │   │  MULTI   │   │  RRF/α   │   │ CrossEnc │   │          │ │
│  │          │   │  INDEX   │   │          │   │ (opt)    │   │ top_k    │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Performance Targets

| Operation | Target Latency | Target Throughput |
|-----------|---------------|-------------------|
| Ingestion (per doc) | < 500ms | 100 docs/min |
| Chunking | < 10ms | 1000 chunks/sec |
| Embedding (batch) | < 1s | 32 chunks/batch |
| Vector search | < 50ms | P99 |
| BM25 search | < 20ms | P99 |
| Hybrid retrieval | < 100ms | P99 |
| With reranking | < 300ms | P99 |

---

## 9. VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-11 | Initial architecture specification |

---

*"The best RAG system is the one you can debug."*
*- ReasonKit Engineering*
