//! Retrieval module for ReasonKit Core
//!
//! Provides hybrid search combining dense (vector) and sparse (BM25) retrieval,
//! with optional cross-encoder reranking for improved precision.
//!
//! ## Components
//!
//! - **HybridRetriever**: Combines BM25 and vector search with RRF fusion
//! - **ExpansionEngine**: Query expansion for better recall
//! - **FusionEngine**: Reciprocal Rank Fusion for combining result sets
//! - **Reranker**: Cross-encoder reranking for precision improvement
//!
//! ## Research Foundation
//!
//! - **RRF Fusion**: Cormack et al. 2009 - "Reciprocal Rank Fusion"
//! - **Cross-Encoder**: Nogueira et al. 2020 - arXiv:2010.06467

pub mod expansion;
pub mod fusion;
pub mod rerank;

pub use expansion::{ExpansionConfig, ExpansionEngine, MultiQueryStrategy};
pub use fusion::{FusedResult, FusionEngine, FusionStrategy, RankedResult};
pub use rerank::{
    CrossEncoderBackend, HeuristicCrossEncoder, RerankStats, RerankedResult, Reranker,
    RerankerCandidate, RerankerConfig,
};

use crate::{
    embedding::EmbeddingPipeline,
    indexing::IndexManager,
    raptor::{RaptorStats, RaptorTree},
    storage::{AccessContext, AccessLevel, Storage},
    Document, Error, MatchSource, Result, RetrievalConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Chunk metadata for fusion reconstruction
#[derive(Debug, Clone)]
struct ChunkMetadata {
    doc_id: Uuid,
    text: String,
    sparse_score: Option<f32>,
    dense_score: Option<f32>,
    section: Option<String>,
}

/// Hybrid retriever combining vector and BM25 search
pub struct HybridRetriever {
    storage: Storage,
    index: IndexManager,
    config: RetrievalConfig,
    raptor_tree: Option<RaptorTree>,
    embedding_pipeline: Option<Arc<EmbeddingPipeline>>,
}

impl HybridRetriever {
    /// Create admin access context for internal operations
    fn admin_context(&self, operation: &str) -> AccessContext {
        AccessContext::new(
            "system".to_string(),
            AccessLevel::Admin,
            operation.to_string(),
        )
    }

    /// Create a new hybrid retriever with in-memory backends
    pub fn in_memory() -> Result<Self> {
        Ok(Self {
            storage: Storage::in_memory(),
            index: IndexManager::in_memory()?,
            config: RetrievalConfig::default(),
            raptor_tree: None,
            embedding_pipeline: None,
        })
    }

    /// Create a new hybrid retriever with the given storage and index
    pub fn new(storage: Storage, index: IndexManager) -> Self {
        Self {
            storage,
            index,
            config: RetrievalConfig::default(),
            raptor_tree: None,
            embedding_pipeline: None,
        }
    }

    /// Set the retrieval configuration
    pub fn with_config(mut self, config: RetrievalConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the embedding pipeline
    pub fn with_embedding_pipeline(mut self, pipeline: Arc<EmbeddingPipeline>) -> Self {
        self.embedding_pipeline = Some(pipeline);
        self
    }

    /// Get the storage backend
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Get the index manager
    pub fn index(&self) -> &IndexManager {
        &self.index
    }

    /// Get the embedding pipeline
    pub fn embedding_pipeline(&self) -> Option<&Arc<EmbeddingPipeline>> {
        self.embedding_pipeline.as_ref()
    }

    /// Index a document for retrieval
    pub async fn add_document(&self, doc: &Document) -> Result<()> {
        // Store the document
        let context = self.admin_context("add_document");
        self.storage.store_document(doc, &context).await?;

        // Index in BM25
        self.index.index_document(doc)?;

        // Generate and store embeddings if pipeline is configured
        if let Some(ref pipeline) = self.embedding_pipeline {
            if !doc.chunks.is_empty() {
                let embeddings = pipeline.embed_chunks(&doc.chunks).await?;

                // Store embeddings in vector database
                for (chunk, embedding_result) in doc.chunks.iter().zip(embeddings.iter()) {
                    if let Some(ref embedding) = embedding_result.dense {
                        self.storage
                            .store_embeddings(&chunk.id, embedding, &context)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Index a document with pre-computed embeddings
    pub async fn add_document_with_embeddings(
        &self,
        doc: &Document,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        if doc.chunks.len() != embeddings.len() {
            return Err(Error::embedding(format!(
                "Chunk count ({}) does not match embedding count ({})",
                doc.chunks.len(),
                embeddings.len()
            )));
        }

        // Store the document
        let context = self.admin_context("add_document_with_embeddings");
        self.storage.store_document(doc, &context).await?;

        // Index in BM25
        self.index.index_document(doc)?;

        // Store embeddings
        for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
            self.storage
                .store_embeddings(&chunk.id, embedding, &context)
                .await?;
        }

        Ok(())
    }

    /// Delete a document
    pub async fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        let context = self.admin_context("delete_document");
        self.storage.delete_document(doc_id, &context).await?;
        // Remove from BM25 index as well
        self.index.delete_document(doc_id)?;
        Ok(())
    }

    /// Get retrieval statistics
    pub async fn stats(&self) -> Result<RetrievalStats> {
        let context = self.admin_context("stats");
        let storage_stats = self.storage.stats(&context).await?;
        let index_stats = self.index.stats()?;

        let raptor_stats = self.raptor_tree.as_ref().map(|tree| tree.stats());

        Ok(RetrievalStats {
            document_count: storage_stats.document_count,
            chunk_count: storage_stats.chunk_count,
            indexed_chunks: index_stats.chunk_count,
            embedding_count: storage_stats.embedding_count,
            storage_bytes: storage_stats.size_bytes,
            index_bytes: index_stats.size_bytes,
            raptor_stats,
        })
    }

    /// Enable RAPTOR tree with given configuration
    pub fn with_raptor(mut self, max_depth: usize, cluster_size: usize) -> Self {
        self.raptor_tree = Some(RaptorTree::new(max_depth, cluster_size));
        self
    }

    /// Build RAPTOR tree from current documents
    pub async fn build_raptor_tree(
        &mut self,
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<()> {
        if self.raptor_tree.is_some() {
            // Get all chunks from documents first
            let context = self.admin_context("build_raptor_tree");
            let mut all_chunks = Vec::new();
            let doc_ids = self.storage.list_documents(&context).await?;

            for doc_id in doc_ids {
                if let Some(doc) = self.storage.get_document(&doc_id, &context).await? {
                    all_chunks.extend(doc.chunks);
                }
            }

            // Now build the tree
            if let Some(ref mut tree) = self.raptor_tree {
                tree.build_from_chunks(&all_chunks, embedder, summarizer)
                    .await?;
            }
        }

        Ok(())
    }

    /// Search using hybrid retrieval (vector + BM25)
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        let config = RetrievalConfig {
            top_k,
            ..Default::default()
        };
        self.search_hybrid(query, None, &config).await
    }

    /// Search using sparse retrieval (BM25 only)
    pub async fn search_sparse(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        let sparse_results = self.index.search_bm25(query, top_k)?;

        let results = sparse_results
            .into_iter()
            .map(|result| HybridResult {
                doc_id: result.doc_id,
                chunk_id: result.chunk_id,
                text: result.text,
                score: result.score,
                dense_score: None,
                sparse_score: Some(result.score),
                match_source: MatchSource::Sparse,
                section: result.section,
            })
            .collect();

        Ok(results)
    }

    /// Search using dense retrieval (vector only)
    pub async fn search_dense(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        // Generate query embedding
        let embedding = if let Some(ref pipeline) = self.embedding_pipeline {
            pipeline.embed_text(query).await?
        } else {
            return Err(Error::retrieval(
                "Embedding pipeline not configured. Call with_embedding_pipeline() first.",
            ));
        };

        // Search using vector
        let dense_results = self
            .storage
            .search_by_vector(&embedding, top_k, &self.admin_context("search_dense"))
            .await?;

        // Try to enrich results with BM25 metadata when available
        let mut results = Vec::with_capacity(dense_results.len());
        for (chunk_id, score) in dense_results {
            let (doc_id, text, section) = self
                .index
                .get_chunk_by_id(&chunk_id)
                .map(|chunk_info| (chunk_info.doc_id, chunk_info.text, chunk_info.section))
                .unwrap_or_else(|| (Uuid::nil(), String::new(), None));

            results.push(HybridResult {
                doc_id,
                chunk_id,
                text,
                score,
                dense_score: Some(score),
                sparse_score: None,
                match_source: MatchSource::Dense,
                section,
            });
        }

        Ok(results)
    }

    /// Search using hybrid retrieval with custom configuration
    ///
    /// Uses the configured fusion strategy (RRF by default) to combine
    /// dense and sparse retrieval results.
    pub async fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        config: &RetrievalConfig,
    ) -> Result<Vec<HybridResult>> {
        let mut method_results: HashMap<String, Vec<RankedResult>> = HashMap::new();

        // Mapping from chunk_id to full result metadata
        let mut chunk_metadata: HashMap<Uuid, ChunkMetadata> = HashMap::new();

        // Get sparse results (BM25) if alpha allows
        if config.alpha < 1.0 {
            if let Ok(sparse_results) = self.index.search_bm25(query, config.top_k) {
                let ranked_results: Vec<RankedResult> = sparse_results
                    .iter()
                    .enumerate()
                    .map(|(rank, result)| {
                        // Store metadata for later reconstruction
                        chunk_metadata.insert(
                            result.chunk_id,
                            ChunkMetadata {
                                doc_id: result.doc_id,
                                text: result.text.clone(),
                                sparse_score: Some(result.score),
                                dense_score: None,
                                section: result.section.clone(),
                            },
                        );

                        RankedResult {
                            id: result.chunk_id,
                            score: result.score,
                            rank,
                        }
                    })
                    .collect();

                method_results.insert("sparse".to_string(), ranked_results);
            }
        }

        // Get dense results (vector search) if alpha allows
        if config.alpha > 0.0 {
            // Generate or use provided embedding
            let embedding = if let Some(emb) = query_embedding {
                emb.to_vec()
            } else if let Some(ref pipeline) = self.embedding_pipeline {
                // FIXED: Use embedding pipeline instead of placeholder
                pipeline.embed_text(query).await?
            } else {
                return Err(Error::retrieval(
                    "No query embedding provided and no embedding pipeline configured. \
                     Either provide query_embedding or call with_embedding_pipeline().",
                ));
            };

            if let Ok(dense_results) = self
                .storage
                .search_by_vector(
                    &embedding,
                    config.top_k,
                    &self.admin_context("search_hybrid"),
                )
                .await
            {
                let ranked_results: Vec<RankedResult> = dense_results
                    .iter()
                    .enumerate()
                    .map(|(rank, (chunk_id, score))| {
                        // Update or insert metadata
                        chunk_metadata
                            .entry(*chunk_id)
                            .and_modify(|meta| meta.dense_score = Some(*score))
                            .or_insert_with(|| ChunkMetadata {
                                doc_id: Uuid::nil(), // Will be filled if needed
                                text: String::new(), // Will be filled if needed
                                sparse_score: None,
                                dense_score: Some(*score),
                                section: None,
                            });

                        RankedResult {
                            id: *chunk_id,
                            score: *score,
                            rank,
                        }
                    })
                    .collect();

                method_results.insert("dense".to_string(), ranked_results);
            }
        }

        // If no results from either method, return empty
        if method_results.is_empty() {
            return Ok(Vec::new());
        }

        // Fuse results using default strategy (RRF)
        let fusion_engine = FusionEngine::new(FusionStrategy::default());
        let fused_results = fusion_engine.fuse(method_results)?;

        // Convert fused results to HybridResult
        let mut hybrid_results = Vec::new();
        for fused in fused_results.into_iter().take(config.top_k) {
            if let Some(meta) = chunk_metadata.get(&fused.id) {
                // Determine match source
                let match_source = match (meta.dense_score, meta.sparse_score) {
                    (Some(_), Some(_)) => MatchSource::Hybrid,
                    (Some(_), None) => MatchSource::Dense,
                    (None, Some(_)) => MatchSource::Sparse,
                    (None, None) => MatchSource::Hybrid, // Fallback
                };

                hybrid_results.push(HybridResult {
                    doc_id: meta.doc_id,
                    chunk_id: fused.id,
                    text: meta.text.clone(),
                    score: fused.fusion_score,
                    dense_score: meta.dense_score,
                    sparse_score: meta.sparse_score,
                    match_source,
                    section: meta.section.clone(),
                });
            }
        }

        Ok(hybrid_results)
    }
}

/// Result from hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// Document ID
    pub doc_id: Uuid,
    /// Chunk ID
    pub chunk_id: Uuid,
    /// Chunk text
    pub text: String,
    /// Combined score
    pub score: f32,
    /// Dense (vector) score
    pub dense_score: Option<f32>,
    /// Sparse (BM25) score
    pub sparse_score: Option<f32>,
    /// Match source
    pub match_source: MatchSource,
    /// Section name (if available)
    pub section: Option<String>,
}

/// Retrieval statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalStats {
    /// Number of documents
    pub document_count: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Number of indexed chunks
    pub indexed_chunks: usize,
    /// Number of embeddings
    pub embedding_count: usize,
    /// Storage size in bytes
    pub storage_bytes: u64,
    /// Index size in bytes
    pub index_bytes: u64,
    /// RAPTOR tree statistics (if enabled)
    pub raptor_stats: Option<RaptorStats>,
}

/// Knowledge base combining all retrieval functionality
pub struct KnowledgeBase {
    /// Underlying hybrid retriever
    pub(crate) retriever: HybridRetriever,
}

impl KnowledgeBase {
    /// Create a new in-memory knowledge base
    pub fn in_memory() -> Result<Self> {
        Ok(Self {
            retriever: HybridRetriever::in_memory()?,
        })
    }

    /// Create with embedding pipeline
    pub fn with_embedding_pipeline(mut self, pipeline: Arc<EmbeddingPipeline>) -> Self {
        self.retriever = self.retriever.with_embedding_pipeline(pipeline);
        self
    }

    /// Add a document to the knowledge base
    pub async fn add(&self, doc: &Document) -> Result<()> {
        self.retriever.add_document(doc).await
    }

    /// Query the knowledge base
    pub async fn query(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        self.retriever.search(query, top_k).await
    }

    /// Query with custom configuration
    pub async fn query_with_config(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<Vec<HybridResult>> {
        self.retriever.search_hybrid(query, None, config).await
    }

    /// Get statistics
    pub async fn stats(&self) -> Result<RetrievalStats> {
        self.retriever.stats().await
    }

    /// Delete a document
    pub async fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        let context = self.retriever.admin_context("delete_document");
        self.retriever
            .storage
            .delete_document(doc_id, &context)
            .await?;
        // Remove from BM25 index as well
        self.retriever.index.delete_document(doc_id)?;
        Ok(())
    }

    /// Get the retriever
    pub fn retriever(&self) -> &HybridRetriever {
        &self.retriever
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, DocumentType, EmbeddingIds, Source, SourceType};
    use chrono::Utc;

    fn create_test_document() -> Document {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.md".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Note, source)
            .with_content("Machine learning and artificial intelligence.".to_string());

        doc.chunks = vec![
            Chunk {
                id: Uuid::new_v4(),
                text: "Machine learning is a subset of artificial intelligence.".to_string(),
                index: 0,
                start_char: 0,
                end_char: 55,
                token_count: Some(10),
                section: Some("Introduction".to_string()),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Neural networks are used for deep learning tasks.".to_string(),
                index: 1,
                start_char: 56,
                end_char: 104,
                token_count: Some(9),
                section: Some("Neural Networks".to_string()),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
        ];

        doc
    }

    #[tokio::test]
    async fn test_knowledge_base() {
        let kb = KnowledgeBase::in_memory().unwrap();
        let doc = create_test_document();

        // Add document
        kb.add(&doc).await.unwrap();

        // Query using sparse search (no embeddings needed)
        // Use the underlying retriever for sparse-only search
        let results = kb
            .retriever
            .search_sparse("machine learning", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.to_lowercase().contains("machine learning"));

        // Stats
        let stats = kb.stats().await.unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 2);
    }

    #[tokio::test]
    async fn test_hybrid_retriever() {
        let retriever = HybridRetriever::in_memory().unwrap();
        let doc = create_test_document();

        // Add document
        retriever.add_document(&doc).await.unwrap();

        // Sparse search
        let results = retriever.search_sparse("neural networks", 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].match_source, MatchSource::Sparse);
    }
}
