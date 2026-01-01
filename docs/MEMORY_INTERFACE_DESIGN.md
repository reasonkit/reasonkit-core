# Memory Interface Trait Design

## Overview

The `MemoryService` trait defines how **reasonkit-core** interfaces with **reasonkit-mem**. It provides a clean, async-first abstraction layer for:

- Document storage and retrieval
- Hybrid search (dense vector + sparse BM25)
- Context assembly for LLM reasoning
- Vector embeddings and similarity search
- Index management and health monitoring

## Design Philosophy

### Core Principles

1. **Async-First**: All operations are async using `async_trait` and tokio
2. **Trait-Based**: Multiple implementations possible (Qdrant, file, in-memory, etc.)
3. **Result-Oriented**: All operations return `Result<T>` with structured error handling
4. **Batch-Friendly**: Efficient bulk operations for documents and embeddings
5. **Type-Safe**: Full type checking with serde serialization support

### Why This Design?

- **Decoupling**: reasonkit-core doesn't depend directly on reasonkit-mem internals
- **Testability**: Easy to mock for unit tests
- **Extensibility**: Custom implementations can be provided
- **Performance**: Allows efficient batch operations and caching
- **Flexibility**: Works with different storage backends transparently

## Trait Structure

### 1. Document Storage Operations

```rust
pub trait MemoryService: Send + Sync {
    // Single document
    async fn store_document(&self, document: &Document) -> MemoryResult<Uuid>;
    async fn get_document(&self, doc_id: &Uuid) -> MemoryResult<Option<Document>>;
    async fn delete_document(&self, doc_id: &Uuid) -> MemoryResult<()>;
    async fn list_documents(&self) -> MemoryResult<Vec<Uuid>>;

    // Batch documents
    async fn store_documents(&self, documents: &[Document]) -> MemoryResult<Vec<Uuid>>;
}
```

**Use Cases:**
- Loading research papers or documentation
- Versioning document updates
- Removing outdated content
- Bulk ingestion during initialization

### 2. Search & Retrieval Operations

```rust
// Hybrid search (default, recommended)
async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>>;
async fn search_with_config(&self, query: &str, config: &ContextConfig)
    -> MemoryResult<Vec<SearchResult>>;

// Dense only (semantic similarity)
async fn search_by_vector(&self, embedding: &[f32], top_k: usize)
    -> MemoryResult<Vec<SearchResult>>;

// Sparse only (keyword matching)
async fn search_by_keywords(&self, query: &str, top_k: usize)
    -> MemoryResult<Vec<SearchResult>>;
```

**Search Methods Explained:**

| Method | Input | Use Case | Speed | Relevance |
|--------|-------|----------|-------|-----------|
| `search()` | Text query | General purpose | Fast | Good (semantic + keyword) |
| `search_with_config()` | Text + config | Fine-tuned retrieval | Variable | Excellent |
| `search_by_vector()` | Pre-computed embedding | Pre-embedded queries | Very Fast | Good (semantic only) |
| `search_by_keywords()` | Text keywords | Exact term matching | Very Fast | Variable |

**Hybrid Search Details:**
- Combines dense vectors (semantic meaning) and sparse BM25 (keyword matching)
- Uses Reciprocal Rank Fusion (RRF) to combine results
- Optional cross-encoder reranking for precision
- Configurable alpha parameter (0=sparse only, 1=dense only)

### 3. Context Assembly Operations

```rust
// Primary method for reasoning
async fn get_context(&self, query: &str, top_k: usize) -> MemoryResult<ContextWindow>;
async fn get_context_with_config(&self, query: &str, config: &ContextConfig)
    -> MemoryResult<ContextWindow>;

// Access chunks directly
async fn get_document_chunks(&self, doc_id: &Uuid) -> MemoryResult<Vec<Chunk>>;
```

**Context Window Structure:**
```rust
pub struct ContextWindow {
    pub chunks: Vec<Chunk>,           // Ordered relevant chunks
    pub documents: Vec<Document>,     // Associated documents
    pub scores: Vec<f32>,             // Relevance scores
    pub sources: Vec<MatchSource>,    // Dense/Sparse/Hybrid/Raptor
    pub token_count: usize,           // Approximate total tokens
    pub quality: ContextQuality,      // Diversity, coverage metrics
}
```

**Example Usage:**
```rust
let context = memory.get_context("What is RAG?", 5).await?;

// Assemble prompt
let prompt = format!(
    "Context:\n{}\n\nQuestion: {}",
    context.chunks.iter()
        .map(|c| &c.text)
        .collect::<Vec<_>>()
        .join("\n---\n"),
    "Why is RAG important?"
);
```

### 4. Embedding Operations

```rust
async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>>;
async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>>;
```

**Features:**
- Uses configured embedding model (local ONNX or remote API)
- Results cached where possible
- Batch operations optimized for throughput

### 5. Indexing Operations

```rust
async fn build_indexes(&self) -> MemoryResult<()>;
async fn rebuild_indexes(&self) -> MemoryResult<()>;
async fn check_index_health(&self) -> MemoryResult<IndexStats>;
```

**When to Use:**
- `build_indexes()`: After adding new documents (idempotent)
- `rebuild_indexes()`: After corruption or for optimization
- `check_index_health()`: Monitor index consistency

### 6. Stats & Health Monitoring

```rust
async fn stats(&self) -> MemoryResult<MemoryStats>;
async fn is_healthy(&self) -> MemoryResult<bool>;
```

**Returned Metrics:**
```rust
pub struct MemoryStats {
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedding_count: usize,
    pub storage_size_bytes: u64,
    pub indexed_count: usize,
    pub is_healthy: bool,
}
```

## Configuration: ContextConfig

Fine-tune retrieval behavior for specific use cases:

```rust
pub struct ContextConfig {
    pub top_k: usize,              // Number of chunks (default: 10)
    pub min_score: f32,            // Minimum relevance (default: 0.0)
    pub alpha: f32,                // Hybrid balance (default: 0.7)
    pub use_raptor: bool,          // Hierarchical tree (default: false)
    pub rerank: bool,              // Cross-encoder (default: false)
    pub include_metadata: bool,    // Metadata in results (default: true)
}
```

**Optimization Examples:**

**Fast, Default Search:**
```rust
ContextConfig {
    top_k: 10,
    alpha: 0.7,
    use_raptor: false,
    rerank: false,
    ..Default::default()
}
```

**High-Quality, Precision Search:**
```rust
ContextConfig {
    top_k: 20,
    min_score: 0.5,
    alpha: 0.7,    // Balanced hybrid
    use_raptor: true,  // Hierarchical retrieval
    rerank: true,      // Cross-encoder precision
    ..Default::default()
}
```

**Semantic-Only Search:**
```rust
ContextConfig {
    alpha: 1.0,    // Dense only
    ..Default::default()
}
```

## Error Handling

Structured error types for better diagnostics:

```rust
pub struct MemoryError {
    pub category: ErrorCategory,
    pub message: String,
    pub context: Option<String>,
}

pub enum ErrorCategory {
    Storage,      // Persistence failed
    Embedding,    // Vector computation failed
    Retrieval,    // Search failed
    Indexing,     // Index operation failed
    NotFound,     // Document/chunk not found
    InvalidInput, // Bad data
    Config,       // Configuration error
    Internal,     // Unexpected error
}
```

## Thread Safety

All implementations MUST be:

1. **Send + Sync**: Safe to share across threads
2. **Internally Synchronized**: Use Arc<RwLock<T>>, channels, or equivalent
3. **Panic-Safe**: Errors propagate, never panic

```rust
#[async_trait]
pub trait MemoryService: Send + Sync {
    // ... all methods
}
```

## Implementation Example Structure

```rust
pub struct MyMemoryImpl {
    storage: Arc<RwLock<...>>,
    index: Arc<RwLock<...>>,
    embedding: Arc<EmbeddingService>,
}

#[async_trait]
impl MemoryService for MyMemoryImpl {
    async fn store_document(&self, document: &Document) -> MemoryResult<Uuid> {
        let mut storage = self.storage.write().await;
        // Store logic...
        Ok(doc_id)
    }

    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>> {
        let embedding = self.embedding.embed(query).await?;
        let storage = self.storage.read().await;
        // Search logic...
        Ok(results)
    }

    // ... implement other methods
}
```

## Integration with reasonkit-mem

The trait bridges reasonkit-core and reasonkit-mem:

```
reasonkit-core
    ↓
MemoryService trait
    ↓
reasonkit-mem implementations
    ├── Storage (Qdrant + File)
    ├── Embeddings (ONNX + APIs)
    ├── Retrieval (Hybrid + RRF)
    ├── Indexing (Tantivy BM25)
    └── RAPTOR Trees
```

## Feature Flag Behavior

### With `memory` feature enabled:
```rust
pub use reasonkit_mem::{
    Chunk, Document, DocumentContent, DocumentType,
    MatchSource, Metadata, ProcessingState,
    ProcessingStatus, RetrievalConfig, SearchResult,
    Source, SourceType
};
```

### Without `memory` feature:
```rust
pub type Chunk = ();
pub type Document = ();
// ... type stubs allow code to compile
// implementations would need custom types
```

## Performance Considerations

1. **Batch Operations**: Use `store_documents()` and `embed_batch()` for bulk processing
2. **Caching**: Embeddings cached by default to avoid recomputation
3. **Lazy Indexing**: Index building can be deferred until needed
4. **Connection Pooling**: Qdrant backend uses connection pools
5. **Access Levels**: Fine-grained access control available via AccessContext

## Testing Strategy

Mock implementation for testing:

```rust
pub struct MockMemoryService {
    docs: Arc<RwLock<HashMap<Uuid, Document>>>,
    chunks: Arc<RwLock<Vec<Chunk>>>,
}

#[async_trait]
impl MemoryService for MockMemoryService {
    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid> {
        let id = doc.id;
        self.docs.write().await.insert(id, doc.clone());
        Ok(id)
    }

    // ... other methods
}

#[tokio::test]
async fn test_search() {
    let memory = MockMemoryService::default();
    let results = memory.search("query", 10).await.unwrap();
    assert!(!results.is_empty());
}
```

## API Stability

**Stable (v0.1.0+):**
- Document storage operations
- Hybrid search with ContextConfig
- Context assembly (ContextWindow)
- Embedding operations
- Stats and health checks

**Experimental (may change):**
- RAPTOR tree integration (optimization opportunity)
- Cross-encoder reranking options (quality tuning)
- AccessContext integration (fine-grained permissions)

## Related Documentation

- `reasonkit-mem/README.md`: Storage backend details
- `reasonkit-core/src/rag/mod.rs`: Full RAG pipeline using this trait
- `ORCHESTRATOR.md`: System architecture overview
