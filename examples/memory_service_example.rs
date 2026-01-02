//! Example: MemoryService Trait Usage
//!
//! This example demonstrates comprehensive usage of the MemoryService trait
//! from the traits module. It shows:
//!
//! - Creating mock implementations
//! - Document storage and retrieval
//! - Vector and hybrid search
//! - Embedding operations
//! - Context window assembly
//! - Index management
//! - Error handling patterns
//! - Both sync and async usage patterns
//!
//! # Running this example
//!
//! ```bash
//! cargo run --example memory_service_example
//! ```
//!
//! # Architecture
//!
//! The MemoryService trait defines the contract between reasonkit-core and
//! reasonkit-mem. This allows reasonkit-core to work with any memory backend
//! that implements the trait.
//!
//! ```text
//! reasonkit-core (consumer)
//!        |
//!        v
//! MemoryService trait <-- defined in reasonkit-core/src/traits/memory.rs
//!        ^
//!        |
//! reasonkit-mem (implementation)
//! ```

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// Import from the traits module
use reasonkit::traits::{
    Chunk, ContextWindow, Document, HybridConfig, IndexConfig, IndexStats, MemoryConfig,
    MemoryError, MemoryResult, MemoryService, RetrievalSource, SearchResult,
};

// ============================================================================
// MOCK IMPLEMENTATION
// ============================================================================

/// Mock implementation of MemoryService for demonstration.
///
/// In production, you would use the implementation from reasonkit-mem.
/// This mock is useful for:
/// - Testing without external dependencies
/// - Understanding the trait interface
/// - Prototyping before integration
struct MockMemoryService {
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
    chunks: Arc<RwLock<HashMap<Uuid, Chunk>>>,
    #[allow(dead_code)]
    embeddings: Arc<RwLock<HashMap<Uuid, Vec<f32>>>>,
    config: MemoryConfig,
}

impl MockMemoryService {
    /// Create a new mock memory service with default configuration.
    fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
            documents: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration.
    fn with_config(config: MemoryConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Generate a mock embedding vector.
    fn mock_embedding(&self, _text: &str) -> Vec<f32> {
        // In production, this would call an embedding model
        // For demo, generate random-ish values based on text length
        let dims = self.config.embedding_dimensions;
        (0..dims).map(|i| (i as f32 * 0.01).sin()).collect()
    }

    /// Calculate mock similarity score.
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        // Cosine similarity approximation
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}

#[async_trait]
impl MemoryService for MockMemoryService {
    // -------------------------------------------------------------------------
    // Storage Operations
    // -------------------------------------------------------------------------

    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid> {
        let id = doc.id.unwrap_or_else(Uuid::new_v4);
        let mut doc_clone = doc.clone();
        doc_clone.id = Some(id);
        doc_clone.created_at = Some(chrono::Utc::now().timestamp());

        // Store document
        self.documents
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?
            .insert(id, doc_clone.clone());

        // Create chunks from content
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.chunk_overlap;
        let content = &doc_clone.content;

        let mut chunks_to_store = Vec::new();
        let mut start = 0;

        while start < content.len() {
            let end = (start + chunk_size).min(content.len());
            let chunk_text = &content[start..end];

            let chunk = Chunk {
                id: Some(Uuid::new_v4()),
                document_id: id,
                content: chunk_text.to_string(),
                index: chunks_to_store.len(),
                embedding: Some(self.mock_embedding(chunk_text)),
                metadata: HashMap::new(),
            };
            chunks_to_store.push(chunk);

            start = if end >= content.len() {
                content.len()
            } else {
                end.saturating_sub(overlap)
            };
        }

        // Store chunks
        {
            let mut chunks_lock = self
                .chunks
                .write()
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            for chunk in chunks_to_store {
                if let Some(chunk_id) = chunk.id {
                    chunks_lock.insert(chunk_id, chunk);
                }
            }
        }

        Ok(id)
    }

    async fn store_chunks(&self, chunks: &[Chunk]) -> MemoryResult<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(chunks.len());
        let mut chunks_lock = self
            .chunks
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        for chunk in chunks {
            let id = chunk.id.unwrap_or_else(Uuid::new_v4);
            let mut chunk_clone = chunk.clone();
            chunk_clone.id = Some(id);
            chunks_lock.insert(id, chunk_clone);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn delete_document(&self, id: Uuid) -> MemoryResult<()> {
        // Remove document
        self.documents
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?
            .remove(&id);

        // Remove associated chunks
        let mut chunks_lock = self
            .chunks
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        chunks_lock.retain(|_, chunk| chunk.document_id != id);

        Ok(())
    }

    async fn update_document(&self, id: Uuid, doc: &Document) -> MemoryResult<()> {
        let mut docs = self
            .documents
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        if !docs.contains_key(&id) {
            return Err(MemoryError::NotFound(id));
        }

        let mut updated = doc.clone();
        updated.id = Some(id);
        docs.insert(id, updated);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Retrieval Operations
    // -------------------------------------------------------------------------

    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>> {
        let query_embedding = self.mock_embedding(query);
        let chunks = self
            .chunks
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let mut results: Vec<(f32, Chunk)> = chunks
            .values()
            .filter_map(|chunk| {
                chunk.embedding.as_ref().map(|emb| {
                    let score = self.similarity(&query_embedding, emb);
                    (score, chunk.clone())
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(score, chunk)| SearchResult {
                chunk,
                score,
                source: RetrievalSource::Vector,
            })
            .collect())
    }

    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        config: HybridConfig,
    ) -> MemoryResult<Vec<SearchResult>> {
        // Vector search
        let vector_results = self.search(query, top_k * 2).await?;

        // BM25-style keyword search (simplified)
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let chunks = self
            .chunks
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let mut bm25_scores: HashMap<Uuid, f32> = HashMap::new();
        for chunk in chunks.values() {
            let mut score = 0.0;
            for term in &query_terms {
                if chunk.content.to_lowercase().contains(&term.to_lowercase()) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                if let Some(id) = chunk.id {
                    bm25_scores.insert(id, score / query_terms.len() as f32);
                }
            }
        }

        // Reciprocal Rank Fusion
        let mut fused_scores: HashMap<Uuid, f32> = HashMap::new();
        let k = 60.0; // RRF constant

        for (rank, result) in vector_results.iter().enumerate() {
            if let Some(id) = result.chunk.id {
                let rrf_score = config.vector_weight / (k + rank as f32);
                *fused_scores.entry(id).or_insert(0.0) += rrf_score;
            }
        }

        let mut bm25_sorted: Vec<_> = bm25_scores.into_iter().collect();
        bm25_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (id, _)) in bm25_sorted.into_iter().enumerate() {
            let rrf_score = config.bm25_weight / (k + rank as f32);
            *fused_scores.entry(id).or_insert(0.0) += rrf_score;
        }

        // Collect and sort results
        let mut final_results: Vec<SearchResult> = fused_scores
            .into_iter()
            .filter_map(|(id, score)| {
                chunks.get(&id).map(|chunk| SearchResult {
                    chunk: chunk.clone(),
                    score,
                    source: RetrievalSource::Hybrid,
                })
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let result_count = if config.use_reranker {
            config.reranker_top_k.min(top_k)
        } else {
            top_k
        };

        Ok(final_results.into_iter().take(result_count).collect())
    }

    async fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>> {
        let docs = self
            .documents
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(docs.get(&id).cloned())
    }

    async fn get_context(&self, query: &str, max_tokens: usize) -> MemoryResult<ContextWindow> {
        // Search for relevant chunks
        let results = self.search(query, 20).await?;

        // Estimate tokens (roughly 4 chars per token)
        let mut total_tokens = 0;
        let mut selected_chunks = Vec::new();

        for result in results {
            let chunk_tokens = result.chunk.content.len() / 4;
            if total_tokens + chunk_tokens > max_tokens {
                break;
            }
            total_tokens += chunk_tokens;
            selected_chunks.push(result);
        }

        Ok(ContextWindow {
            chunks: selected_chunks,
            total_tokens,
            truncated: total_tokens >= max_tokens,
        })
    }

    // -------------------------------------------------------------------------
    // Embedding Operations
    // -------------------------------------------------------------------------

    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(self.mock_embedding(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.mock_embedding(t)).collect())
    }

    // -------------------------------------------------------------------------
    // Index Management
    // -------------------------------------------------------------------------

    async fn create_index(&self, config: IndexConfig) -> MemoryResult<()> {
        println!(
            "[MockMemory] Creating index '{}' with {} dimensions, {:?} metric",
            config.name, config.dimensions, config.metric
        );
        Ok(())
    }

    async fn rebuild_index(&self) -> MemoryResult<()> {
        println!("[MockMemory] Rebuilding index...");
        Ok(())
    }

    async fn get_stats(&self) -> MemoryResult<IndexStats> {
        let docs = self
            .documents
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        let chunks = self
            .chunks
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(IndexStats {
            total_documents: docs.len(),
            total_chunks: chunks.len(),
            total_vectors: chunks.values().filter(|c| c.embedding.is_some()).count(),
            index_size_bytes: (chunks.len() * 1024) as u64, // Estimate
            last_updated: chrono::Utc::now().timestamp(),
        })
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    fn config(&self) -> &MemoryConfig {
        &self.config
    }

    fn set_config(&mut self, config: MemoryConfig) {
        self.config = config;
    }

    // -------------------------------------------------------------------------
    // Health & Lifecycle
    // -------------------------------------------------------------------------

    async fn health_check(&self) -> MemoryResult<bool> {
        // Check that locks are not poisoned
        let _documents_guard = self
            .documents
            .read()
            .map_err(|_| MemoryError::Storage("Lock poisoned".to_string()))?;
        let _chunks_guard = self
            .chunks
            .read()
            .map_err(|_| MemoryError::Storage("Lock poisoned".to_string()))?;
        Ok(true)
    }

    async fn flush(&self) -> MemoryResult<()> {
        println!("[MockMemory] Flushing to storage...");
        Ok(())
    }

    async fn shutdown(&self) -> MemoryResult<()> {
        println!("[MockMemory] Shutting down...");
        Ok(())
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a sample document for testing.
fn create_sample_document(content: &str, source: &str) -> Document {
    Document {
        id: None,
        content: content.to_string(),
        metadata: HashMap::from([
            ("source".to_string(), source.to_string()),
            ("type".to_string(), "example".to_string()),
        ]),
        source: Some(source.to_string()),
        created_at: None,
    }
}

/// Print search results in a formatted way.
fn print_results(results: &[SearchResult]) {
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. [Score: {:.4}] [Source: {:?}] {}...",
            i + 1,
            result.score,
            result.source,
            &result.chunk.content[..result.chunk.content.len().min(60)]
        );
    }
}

// ============================================================================
// ASYNC USAGE PATTERNS
// ============================================================================

/// Demonstrates async document lifecycle operations.
async fn demo_document_lifecycle(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Document Lifecycle Demo ---\n");

    // 1. Store documents
    println!("1. Storing documents...");
    let doc1 = create_sample_document(
        "Rust is a systems programming language focused on safety, speed, and concurrency. \
         It achieves memory safety without garbage collection through its ownership system.",
        "rust-docs",
    );
    let doc2 = create_sample_document(
        "Python is a high-level programming language known for its simplicity and readability. \
         It supports multiple programming paradigms including procedural, object-oriented, and functional.",
        "python-docs"
    );

    let id1 = memory.store_document(&doc1).await?;
    let id2 = memory.store_document(&doc2).await?;
    println!("   Stored: {} and {}", id1, id2);

    // 2. Retrieve document
    println!("\n2. Retrieving document...");
    if let Some(retrieved) = memory.get_by_id(id1).await? {
        println!(
            "   Found: {} chars, source: {:?}",
            retrieved.content.len(),
            retrieved.source
        );
    }

    // 3. Update document
    println!("\n3. Updating document...");
    let updated_doc = create_sample_document(
        "Rust is a modern systems programming language. Updated content.",
        "rust-docs-v2",
    );
    memory.update_document(id1, &updated_doc).await?;
    println!("   Updated document {}", id1);

    // 4. Get stats
    println!("\n4. Index statistics...");
    let stats = memory.get_stats().await?;
    println!("   Documents: {}", stats.total_documents);
    println!("   Chunks: {}", stats.total_chunks);
    println!("   Vectors: {}", stats.total_vectors);

    // 5. Delete document
    println!("\n5. Deleting document...");
    memory.delete_document(id2).await?;
    println!("   Deleted document {}", id2);

    // Verify deletion
    let stats = memory.get_stats().await?;
    println!("   Remaining documents: {}", stats.total_documents);

    Ok(())
}

/// Demonstrates various search patterns.
async fn demo_search_patterns(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Search Patterns Demo ---\n");

    // Add some documents for searching
    let documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and make decisions based on visual data.",
        "Reinforcement learning trains agents through reward signals in an environment.",
    ];

    println!("1. Adding {} documents...", documents.len());
    for (i, content) in documents.iter().enumerate() {
        let doc = create_sample_document(content, &format!("ml-doc-{}", i));
        memory.store_document(&doc).await?;
    }

    // 2. Vector search
    println!("\n2. Vector search for 'neural networks'...");
    let results = memory.search("neural networks", 3).await?;
    print_results(&results);

    // 3. Hybrid search
    println!("\n3. Hybrid search with custom weights...");
    let hybrid_config = HybridConfig {
        vector_weight: 0.6,
        bm25_weight: 0.4,
        use_reranker: true,
        reranker_top_k: 5,
    };
    let results = memory
        .hybrid_search("learning from data", 3, hybrid_config)
        .await?;
    print_results(&results);

    // 4. Context window
    println!("\n4. Building context window (max 100 tokens)...");
    let context = memory.get_context("What is AI?", 100).await?;
    println!("   Chunks: {}", context.chunks.len());
    println!("   Tokens: {}", context.total_tokens);
    println!("   Truncated: {}", context.truncated);

    Ok(())
}

/// Demonstrates embedding operations.
async fn demo_embeddings(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Embedding Demo ---\n");

    // 1. Single embedding
    println!("1. Generating single embedding...");
    let embedding = memory.embed("Hello, world!").await?;
    println!("   Dimensions: {}", embedding.len());
    println!(
        "   First 5 values: {:?}",
        &embedding[..5.min(embedding.len())]
    );

    // 2. Batch embedding
    println!("\n2. Generating batch embeddings...");
    let texts = &["First text", "Second text", "Third text"];
    let embeddings = memory.embed_batch(texts).await?;
    println!("   Generated {} embeddings", embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        println!("   [{}] {} dimensions", i, emb.len());
    }

    Ok(())
}

/// Demonstrates index management.
async fn demo_index_management(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Index Management Demo ---\n");

    // 1. Create index
    println!("1. Creating index...");
    let index_config = IndexConfig {
        name: "my-index".to_string(),
        dimensions: 384,
        metric: reasonkit::traits::DistanceMetric::Cosine,
        ef_construction: 200,
        m: 16,
    };
    memory.create_index(index_config).await?;

    // 2. Rebuild index
    println!("\n2. Rebuilding index...");
    memory.rebuild_index().await?;

    // 3. Get stats
    println!("\n3. Getting index statistics...");
    let stats = memory.get_stats().await?;
    println!("   Index size: {} bytes", stats.index_size_bytes);
    println!("   Last updated: {}", stats.last_updated);

    Ok(())
}

/// Demonstrates health and lifecycle operations.
async fn demo_health_lifecycle(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Health & Lifecycle Demo ---\n");

    // 1. Health check
    println!("1. Health check...");
    let healthy = memory.health_check().await?;
    println!(
        "   Status: {}",
        if healthy { "Healthy" } else { "Unhealthy" }
    );

    // 2. Configuration
    println!("\n2. Current configuration...");
    let config = memory.config();
    println!("   Chunk size: {}", config.chunk_size);
    println!("   Chunk overlap: {}", config.chunk_overlap);
    println!("   Embedding model: {}", config.embedding_model);
    println!("   Embedding dimensions: {}", config.embedding_dimensions);
    println!("   Max context tokens: {}", config.max_context_tokens);

    // 3. Flush
    println!("\n3. Flushing to storage...");
    memory.flush().await?;

    Ok(())
}

/// Demonstrates error handling patterns.
async fn demo_error_handling(memory: &impl MemoryService) -> MemoryResult<()> {
    println!("\n--- Error Handling Demo ---\n");

    // 1. Not found error
    println!("1. Handling NotFound error...");
    let fake_id = Uuid::new_v4();
    match memory.get_by_id(fake_id).await {
        Ok(Some(doc)) => println!("   Found: {:?}", doc.source),
        Ok(None) => println!("   Document not found (expected behavior)"),
        Err(e) => println!("   Error: {}", e),
    }

    // 2. Update non-existent document
    println!("\n2. Updating non-existent document...");
    let doc = create_sample_document("test", "test");
    match memory.update_document(fake_id, &doc).await {
        Ok(_) => println!("   Updated successfully (unexpected)"),
        Err(MemoryError::NotFound(id)) => println!("   NotFound error: {} (expected)", id),
        Err(e) => println!("   Other error: {}", e),
    }

    Ok(())
}

// ============================================================================
// SYNCHRONOUS WRAPPER PATTERN
// ============================================================================

/// Demonstrates how to use async MemoryService from synchronous code.
///
/// This pattern is useful when integrating with synchronous codebases
/// or when you need to call async methods from a sync context.
mod sync_wrapper {
    use super::*;
    use tokio::runtime::Runtime;

    /// Synchronous wrapper around MemoryService.
    pub struct SyncMemoryService<M: MemoryService> {
        inner: M,
        runtime: Runtime,
    }

    impl<M: MemoryService> SyncMemoryService<M> {
        /// Create a new synchronous wrapper.
        pub fn new(service: M) -> Self {
            Self {
                inner: service,
                runtime: Runtime::new().expect("Failed to create Tokio runtime"),
            }
        }

        /// Store a document synchronously.
        pub fn store_document(&self, doc: &Document) -> MemoryResult<Uuid> {
            self.runtime.block_on(self.inner.store_document(doc))
        }

        /// Search synchronously.
        pub fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>> {
            self.runtime.block_on(self.inner.search(query, top_k))
        }

        /// Get document by ID synchronously.
        #[allow(dead_code)]
        pub fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>> {
            self.runtime.block_on(self.inner.get_by_id(id))
        }

        /// Embed text synchronously.
        #[allow(dead_code)]
        pub fn embed(&self, text: &str) -> MemoryResult<Vec<f32>> {
            self.runtime.block_on(self.inner.embed(text))
        }

        /// Get stats synchronously.
        pub fn get_stats(&self) -> MemoryResult<IndexStats> {
            self.runtime.block_on(self.inner.get_stats())
        }
    }

    /// Demonstrate synchronous usage.
    pub fn demo_sync_usage() {
        println!("\n--- Synchronous Wrapper Demo ---\n");

        let async_service = MockMemoryService::new();
        let sync_service = SyncMemoryService::new(async_service);

        // Store document (sync)
        println!("1. Storing document (sync)...");
        let doc =
            create_sample_document("This is a test document stored synchronously.", "sync-test");
        match sync_service.store_document(&doc) {
            Ok(id) => println!("   Stored with ID: {}", id),
            Err(e) => println!("   Error: {}", e),
        }

        // Search (sync)
        println!("\n2. Searching (sync)...");
        match sync_service.search("test document", 3) {
            Ok(results) => println!("   Found {} results", results.len()),
            Err(e) => println!("   Error: {}", e),
        }

        // Get stats (sync)
        println!("\n3. Getting stats (sync)...");
        match sync_service.get_stats() {
            Ok(stats) => println!(
                "   Documents: {}, Chunks: {}",
                stats.total_documents, stats.total_chunks
            ),
            Err(e) => println!("   Error: {}", e),
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> MemoryResult<()> {
    println!("=======================================================");
    println!("     ReasonKit MemoryService Trait Usage Example");
    println!("=======================================================");

    // Create mock memory service
    let config = MemoryConfig {
        chunk_size: 200,
        chunk_overlap: 20,
        embedding_model: "mock-model".to_string(),
        embedding_dimensions: 128,
        max_context_tokens: 2048,
        storage_path: None,
    };
    let memory = MockMemoryService::with_config(config);

    // Run async demos
    demo_document_lifecycle(&memory).await?;
    demo_search_patterns(&memory).await?;
    demo_embeddings(&memory).await?;
    demo_index_management(&memory).await?;
    demo_health_lifecycle(&memory).await?;
    demo_error_handling(&memory).await?;

    // Shutdown
    println!("\n--- Shutdown ---\n");
    memory.shutdown().await?;

    // Run sync demo
    sync_wrapper::demo_sync_usage();

    println!("\n=======================================================");
    println!("                    Example Complete");
    println!("=======================================================");
    println!("\nKey Takeaways:");
    println!("  1. MemoryService trait provides a unified interface for memory operations");
    println!("  2. All operations are async-first (use tokio runtime)");
    println!("  3. Supports document storage, search, embeddings, and index management");
    println!("  4. Hybrid search combines vector + BM25 with RRF fusion");
    println!("  5. Use SyncMemoryService wrapper for synchronous codebases");
    println!("  6. Error handling uses structured MemoryError enum");
    println!("\nFor production use, see reasonkit-mem crate for real implementations.");

    Ok(())
}
