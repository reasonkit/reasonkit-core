//! Memory Interface Trait
//!
//! This module defines the trait that reasonkit-core uses to interface with reasonkit-mem.
//! It provides a clean abstraction for document storage, retrieval, and context assembly.
//!
//! ## Design Principles
//!
//! - **Async-first**: All operations are async (tokio runtime required)
//! - **Result-oriented**: All operations return `Result<T>` with structured error handling
//! - **Trait-based**: Allows multiple implementations (in-memory, Qdrant, file-based, etc.)
//! - **Batch-friendly**: Supports operations on multiple documents/queries
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use reasonkit::memory_interface::MemoryService;
//! use reasonkit_mem::Document;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Get memory service implementation (from reasonkit-mem)
//!     let memory = create_memory_service().await?;
//!
//!     // Store a document
//!     memory.store_document(doc).await?;
//!
//!     // Search for related content
//!     let results = memory.search("query text", 10).await?;
//!
//!     // Get context for reasoning
//!     let context = memory.get_context("query", 5).await?;
//!
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Re-export reasonkit-mem types for convenience (when memory feature is enabled)
#[cfg(feature = "memory")]
pub use reasonkit_mem::{
    Chunk, Document, DocumentContent, DocumentType, MatchSource, Metadata, ProcessingState,
    ProcessingStatus, RetrievalConfig, SearchResult, Source, SourceType,
};

// Type stubs for when memory feature is disabled
// These allow code to reference types without compilation errors
#[cfg(not(feature = "memory"))]
pub type Chunk = ();
#[cfg(not(feature = "memory"))]
pub type Document = ();
#[cfg(not(feature = "memory"))]
pub type DocumentContent = ();
#[cfg(not(feature = "memory"))]
pub type DocumentType = ();
#[cfg(not(feature = "memory"))]
pub type Metadata = ();
#[cfg(not(feature = "memory"))]
pub type ProcessingState = ();
#[cfg(not(feature = "memory"))]
pub type ProcessingStatus = ();
#[cfg(not(feature = "memory"))]
pub type RetrievalConfig = ();
#[cfg(not(feature = "memory"))]
pub type SearchResult = ();
#[cfg(not(feature = "memory"))]
pub type Source = ();
#[cfg(not(feature = "memory"))]
pub type SourceType = ();

// MatchSource stub
#[cfg(not(feature = "memory"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchSource {
    Dense,
    Sparse,
    Hybrid,
    Raptor,
}

/// Error type for memory interface operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryError {
    /// Error category
    pub category: ErrorCategory,
    /// Human-readable message
    pub message: String,
    /// Optional error context
    pub context: Option<String>,
}

/// Error categories for memory operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Storage operation failed
    Storage,
    /// Embedding/vector operation failed
    Embedding,
    /// Retrieval/search failed
    Retrieval,
    /// Indexing failed
    Indexing,
    /// Document not found
    NotFound,
    /// Invalid input data
    InvalidInput,
    /// Configuration error
    Config,
    /// Unknown or internal error
    Internal,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}: {}{}",
            self.category,
            self.message,
            self.context
                .as_ref()
                .map(|c| format!(" ({})", c))
                .unwrap_or_default()
        )
    }
}

impl std::error::Error for MemoryError {}

/// Result type for memory interface operations
pub type MemoryResult<T> = std::result::Result<T, MemoryError>;

/// Configuration for context retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Number of chunks to retrieve
    pub top_k: usize,
    /// Minimum relevance score (0.0-1.0)
    pub min_score: f32,
    /// Alpha weight for hybrid search (0=sparse only, 1=dense only)
    pub alpha: f32,
    /// Whether to use RAPTOR hierarchical tree
    pub use_raptor: bool,
    /// Whether to rerank results with cross-encoder
    pub rerank: bool,
    /// Include metadata in results
    pub include_metadata: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.0,
            alpha: 0.7, // Favor semantic search
            use_raptor: false,
            rerank: false,
            include_metadata: true,
        }
    }
}

/// A context window retrieved from memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Ordered list of relevant chunks
    pub chunks: Vec<Chunk>,
    /// Associated documents
    pub documents: Vec<Document>,
    /// Relevance scores for each chunk
    pub scores: Vec<f32>,
    /// Source information (dense, sparse, hybrid, raptor)
    pub sources: Vec<MatchSource>,
    /// Total token count (approximate)
    pub token_count: usize,
    /// Quality metrics
    pub quality: ContextQuality,
}

/// Quality metrics for a context window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextQuality {
    /// Average relevance score
    pub avg_score: f32,
    /// Highest relevance score
    pub max_score: f32,
    /// Lowest relevance score
    pub min_score: f32,
    /// Diversity score (0-1, higher = more diverse)
    pub diversity: f32,
    /// Coverage score (0-1, how complete is the context)
    pub coverage: f32,
}

/// Statistics about memory service state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Number of stored documents
    pub document_count: usize,
    /// Number of chunks across all documents
    pub chunk_count: usize,
    /// Number of embeddings stored
    pub embedding_count: usize,
    /// Total storage size in bytes
    pub storage_size_bytes: u64,
    /// Number of indexed documents
    pub indexed_count: usize,
    /// Memory service health status
    pub is_healthy: bool,
}

/// Main trait for memory service operations
///
/// This trait defines the interface that reasonkit-core uses to interact with reasonkit-mem.
/// Implementations handle:
/// - Document storage and retrieval
/// - Vector embeddings and similarity search
/// - Hybrid search (dense + sparse)
/// - Context assembly for reasoning
///
/// # Thread Safety
///
/// All implementations MUST be:
/// - `Send + Sync` for safe cross-thread sharing
/// - Internally synchronized (locks, Arc<RwLock>, etc.)
/// - Panic-safe (errors should propagate, not panic)
#[async_trait]
pub trait MemoryService: Send + Sync {
    // ==================== DOCUMENT STORAGE ====================

    /// Store a document in memory
    ///
    /// This operation:
    /// 1. Validates the document structure
    /// 2. Stores metadata in the document store
    /// 3. Chunks the content (if not already chunked)
    /// 4. Prepares chunks for embedding
    ///
    /// # Arguments
    /// * `document` - The document to store
    ///
    /// # Returns
    /// * `Ok(Uuid)` - The document ID if successful
    /// * `Err(MemoryError)` - If storage fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let doc_id = memory.store_document(document).await?;
    /// println!("Stored document: {}", doc_id);
    /// ```
    async fn store_document(&self, document: &Document) -> MemoryResult<Uuid>;

    /// Store multiple documents (batch operation)
    ///
    /// Stores documents in parallel where possible for efficiency.
    ///
    /// # Arguments
    /// * `documents` - Slice of documents to store
    ///
    /// # Returns
    /// * `Ok(Vec<Uuid>)` - IDs of stored documents
    /// * `Err(MemoryError)` - If any document fails to store
    async fn store_documents(&self, documents: &[Document]) -> MemoryResult<Vec<Uuid>>;

    /// Retrieve a document by ID
    ///
    /// # Arguments
    /// * `doc_id` - The document UUID
    ///
    /// # Returns
    /// * `Ok(Some(Document))` - The document if found
    /// * `Ok(None)` - If document doesn't exist
    /// * `Err(MemoryError)` - If retrieval fails
    async fn get_document(&self, doc_id: &Uuid) -> MemoryResult<Option<Document>>;

    /// Delete a document by ID
    ///
    /// This removes:
    /// - Document metadata
    /// - All associated chunks
    /// - Embeddings for those chunks
    /// - Index entries
    ///
    /// # Arguments
    /// * `doc_id` - The document UUID
    ///
    /// # Returns
    /// * `Ok(())` - If successful (no error if document doesn't exist)
    /// * `Err(MemoryError)` - If deletion fails
    async fn delete_document(&self, doc_id: &Uuid) -> MemoryResult<()>;

    /// List all document IDs in memory
    ///
    /// # Returns
    /// * `Ok(Vec<Uuid>)` - All document IDs
    /// * `Err(MemoryError)` - If listing fails
    async fn list_documents(&self) -> MemoryResult<Vec<Uuid>>;

    // ==================== SEARCH & RETRIEVAL ====================

    /// Search documents using hybrid search
    ///
    /// Performs a combined search across:
    /// - Dense vector search (semantic similarity)
    /// - Sparse BM25 search (keyword matching)
    /// - Reciprocal Rank Fusion for combining results
    /// - Optional cross-encoder reranking
    ///
    /// # Arguments
    /// * `query` - The search query text
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    /// * `Ok(Vec<SearchResult>)` - Ranked search results
    /// * `Err(MemoryError)` - If search fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let results = memory.search("machine learning optimization", 10).await?;
    /// for result in results {
    ///     println!("Score: {}, Document: {}", result.score, result.document_id);
    /// }
    /// ```
    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>>;

    /// Search with advanced configuration
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `config` - Retrieval configuration
    ///
    /// # Returns
    /// * `Ok(Vec<SearchResult>)` - Ranked results
    /// * `Err(MemoryError)` - If search fails
    async fn search_with_config(
        &self,
        query: &str,
        config: &ContextConfig,
    ) -> MemoryResult<Vec<SearchResult>>;

    /// Vector similarity search
    ///
    /// Searches using only dense embeddings (fast, semantic).
    /// Use when you already have embeddings or want pure semantic search.
    ///
    /// # Arguments
    /// * `embedding` - Query vector
    /// * `top_k` - Number of results
    ///
    /// # Returns
    /// * `Ok(Vec<SearchResult>)` - Top K similar chunks
    /// * `Err(MemoryError)` - If search fails
    async fn search_by_vector(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> MemoryResult<Vec<SearchResult>>;

    /// Keyword search (BM25)
    ///
    /// Searches using only sparse BM25 indexing (fast, keyword-based).
    /// Use when you want keyword matching or have specific terms.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `top_k` - Number of results
    ///
    /// # Returns
    /// * `Ok(Vec<SearchResult>)` - Ranked results by BM25 score
    /// * `Err(MemoryError)` - If search fails
    async fn search_by_keywords(
        &self,
        query: &str,
        top_k: usize,
    ) -> MemoryResult<Vec<SearchResult>>;

    // ==================== CONTEXT ASSEMBLY ====================

    /// Get context window for reasoning
    ///
    /// This is the primary method for assembling context for LLM reasoning.
    /// It returns a structured context window with:
    /// - Ranked, relevant chunks
    /// - Associated documents
    /// - Quality metrics
    /// - Token count estimate
    ///
    /// # Arguments
    /// * `query` - The reasoning query/prompt
    /// * `top_k` - Number of chunks to include
    ///
    /// # Returns
    /// * `Ok(ContextWindow)` - Assembled context
    /// * `Err(MemoryError)` - If context assembly fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let context = memory.get_context("How does RAG improve reasoning?", 5).await?;
    /// println!("Context: {} chunks, {} tokens",
    ///     context.chunks.len(),
    ///     context.token_count);
    ///
    /// // Use context in prompt
    /// let prompt = format!("Context:\n{}\n\nQuestion: ...",
    ///     context.chunks.iter()
    ///         .map(|c| &c.text)
    ///         .collect::<Vec<_>>()
    ///         .join("\n---\n"));
    /// ```
    async fn get_context(&self, query: &str, top_k: usize) -> MemoryResult<ContextWindow>;

    /// Get context with advanced configuration
    ///
    /// # Arguments
    /// * `query` - The reasoning query
    /// * `config` - Context retrieval configuration
    ///
    /// # Returns
    /// * `Ok(ContextWindow)` - Assembled context
    /// * `Err(MemoryError)` - If context assembly fails
    async fn get_context_with_config(
        &self,
        query: &str,
        config: &ContextConfig,
    ) -> MemoryResult<ContextWindow>;

    /// Get chunks by document ID
    ///
    /// # Arguments
    /// * `doc_id` - The document UUID
    ///
    /// # Returns
    /// * `Ok(Vec<Chunk>)` - All chunks in the document
    /// * `Err(MemoryError)` - If operation fails
    async fn get_document_chunks(&self, doc_id: &Uuid) -> MemoryResult<Vec<Chunk>>;

    // ==================== EMBEDDINGS ====================

    /// Embed text and get vector representation
    ///
    /// Uses the configured embedding model to convert text to vectors.
    /// Results are cached where possible.
    ///
    /// # Arguments
    /// * `text` - Text to embed
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - The embedding vector
    /// * `Err(MemoryError)` - If embedding fails
    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>>;

    /// Embed multiple texts (batch operation)
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to embed
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<f32>>)` - Embeddings (same order as input)
    /// * `Err(MemoryError)` - If any embedding fails
    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>>;

    // ==================== INDEXING ====================

    /// Build or update indexes
    ///
    /// Triggers indexing for documents that haven't been indexed yet.
    /// Safe to call multiple times (idempotent for already-indexed docs).
    ///
    /// # Returns
    /// * `Ok(())` - If indexing succeeds
    /// * `Err(MemoryError)` - If indexing fails
    async fn build_indexes(&self) -> MemoryResult<()>;

    /// Rebuild all indexes from scratch
    ///
    /// Use when you suspect corruption or want to optimize.
    /// This is slower but ensures consistency.
    ///
    /// # Returns
    /// * `Ok(())` - If rebuild succeeds
    /// * `Err(MemoryError)` - If rebuild fails
    async fn rebuild_indexes(&self) -> MemoryResult<()>;

    /// Check index health and statistics
    ///
    /// # Returns
    /// * `Ok(IndexStats)` - Index statistics
    /// * `Err(MemoryError)` - If health check fails
    async fn check_index_health(&self) -> MemoryResult<IndexStats>;

    // ==================== STATS & HEALTH ====================

    /// Get current memory service statistics
    ///
    /// # Returns
    /// * `Ok(MemoryStats)` - Current statistics
    /// * `Err(MemoryError)` - If stats retrieval fails
    async fn stats(&self) -> MemoryResult<MemoryStats>;

    /// Check if memory service is healthy
    ///
    /// # Returns
    /// * `Ok(true)` - Service is operational
    /// * `Ok(false)` - Service has issues
    /// * `Err(MemoryError)` - If health check fails
    async fn is_healthy(&self) -> MemoryResult<bool>;

    // ==================== ADVANCED FEATURES ====================

    /// Clear all data (for testing)
    ///
    /// WARNING: This is destructive and irreversible in most implementations.
    /// Only use for testing.
    ///
    /// # Returns
    /// * `Ok(())` - If clear succeeds
    /// * `Err(MemoryError)` - If clear fails
    async fn clear_all(&self) -> MemoryResult<()>;
}

/// Index statistics from indexing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of indexed documents
    pub indexed_docs: usize,
    /// Number of indexed chunks
    pub indexed_chunks: usize,
    /// Index size in bytes
    pub index_size_bytes: u64,
    /// Last indexing timestamp (Unix seconds)
    pub last_indexed_at: i64,
    /// Index is valid and consistent
    pub is_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.alpha, 0.7);
        assert!(!config.use_raptor);
    }

    #[test]
    fn test_memory_error_display() {
        let err = MemoryError {
            category: ErrorCategory::NotFound,
            message: "Document not found".to_string(),
            context: Some("doc_id=123".to_string()),
        };
        let display = format!("{}", err);
        assert!(display.contains("NotFound"));
        assert!(display.contains("Document not found"));
    }

    #[test]
    fn test_context_quality_fields() {
        let quality = ContextQuality {
            avg_score: 0.8,
            max_score: 0.95,
            min_score: 0.65,
            diversity: 0.7,
            coverage: 0.85,
        };

        assert!(quality.avg_score < quality.max_score);
        assert!(quality.min_score < quality.avg_score);
        assert!(quality.diversity >= 0.0 && quality.diversity <= 1.0);
        assert!(quality.coverage >= 0.0 && quality.coverage <= 1.0);
    }
}
