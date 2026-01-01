//! Memory service trait for core <-> mem integration.
//!
//! This trait defines the contract between `reasonkit-core` and `reasonkit-mem`.
//! Implementations live in `reasonkit-mem`, consumers live in `reasonkit-core`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Result type for memory operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Errors that can occur during memory operations.
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Document not found: {0}")]
    NotFound(Uuid),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A document to be stored in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Option<Uuid>,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub source: Option<String>,
    pub created_at: Option<i64>,
}

/// A chunk of a document after splitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Option<Uuid>,
    pub document_id: Uuid,
    pub content: String,
    pub index: usize,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

/// A search result from memory retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub source: RetrievalSource,
}

/// Source of the retrieval result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetrievalSource {
    Vector,
    BM25,
    Hybrid,
}

/// Configuration for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    pub vector_weight: f32,
    pub bm25_weight: f32,
    pub use_reranker: bool,
    pub reranker_top_k: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            bm25_weight: 0.3,
            use_reranker: true,
            reranker_top_k: 10,
        }
    }
}

/// A context window assembled from retrieved chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    pub chunks: Vec<SearchResult>,
    pub total_tokens: usize,
    pub truncated: bool,
}

/// Configuration for index creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub ef_construction: usize,
    pub m: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dimensions: 384,
            metric: DistanceMetric::Cosine,
            ef_construction: 200,
            m: 16,
        }
    }
}

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Statistics about the memory index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_vectors: usize,
    pub index_size_bytes: u64,
    pub last_updated: i64,
}

/// Configuration for the memory service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub max_context_tokens: usize,
    pub storage_path: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            embedding_dimensions: 384,
            max_context_tokens: 4096,
            storage_path: None,
        }
    }
}

/// Core abstraction for memory operations.
///
/// This trait is implemented by `reasonkit-mem` and consumed by `reasonkit-core`.
/// It provides a unified interface for document storage, retrieval, and embedding.
///
/// # Example
///
/// ```ignore
/// use reasonkit_core::traits::{MemoryService, Document};
///
/// async fn example(memory: &impl MemoryService) -> MemoryResult<()> {
///     let doc = Document {
///         id: None,
///         content: "Hello, world!".to_string(),
///         metadata: Default::default(),
///         source: Some("example".to_string()),
///         created_at: None,
///     };
///
///     let id = memory.store_document(&doc).await?;
///     let results = memory.search("hello", 5).await?;
///
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait MemoryService: Send + Sync {
    // ─────────────────────────────────────────────────────────────────────────
    // Storage Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Store a document, returning its assigned ID.
    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid>;

    /// Store multiple chunks, returning their assigned IDs.
    async fn store_chunks(&self, chunks: &[Chunk]) -> MemoryResult<Vec<Uuid>>;

    /// Delete a document and all its chunks.
    async fn delete_document(&self, id: Uuid) -> MemoryResult<()>;

    /// Update an existing document.
    async fn update_document(&self, id: Uuid, doc: &Document) -> MemoryResult<()>;

    // ─────────────────────────────────────────────────────────────────────────
    // Retrieval Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Search for relevant chunks using vector similarity.
    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>>;

    /// Search using hybrid retrieval (vector + BM25 with RRF fusion).
    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        config: HybridConfig,
    ) -> MemoryResult<Vec<SearchResult>>;

    /// Get a document by its ID.
    async fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>>;

    /// Get a context window optimized for the query and token budget.
    async fn get_context(
        &self,
        query: &str,
        max_tokens: usize,
    ) -> MemoryResult<ContextWindow>;

    // ─────────────────────────────────────────────────────────────────────────
    // Embedding Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Embed a single text string.
    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>>;

    /// Embed multiple texts in a batch.
    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>>;

    // ─────────────────────────────────────────────────────────────────────────
    // Index Management
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a new index with the given configuration.
    async fn create_index(&self, config: IndexConfig) -> MemoryResult<()>;

    /// Rebuild the index from stored documents.
    async fn rebuild_index(&self) -> MemoryResult<()>;

    /// Get statistics about the current index.
    async fn get_stats(&self) -> MemoryResult<IndexStats>;

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the current configuration.
    fn config(&self) -> &MemoryConfig;

    /// Update the configuration.
    fn set_config(&mut self, config: MemoryConfig);

    // ─────────────────────────────────────────────────────────────────────────
    // Health & Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if the service is healthy and ready.
    async fn health_check(&self) -> MemoryResult<bool>;

    /// Flush any pending writes to storage.
    async fn flush(&self) -> MemoryResult<()>;

    /// Gracefully shutdown the service.
    async fn shutdown(&self) -> MemoryResult<()>;
}
