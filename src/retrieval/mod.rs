//! Retrieval module for ReasonKit Core
//!
//! Provides hybrid search combining dense (vector) and sparse (BM25) retrieval.

use crate::{
    indexing::IndexManager,
    storage::{Storage, AccessContext, AccessLevel},
    raptor::{RaptorTree, RaptorStats},
    Document, RetrievalConfig, Result, MatchSource,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Hybrid retriever combining vector and BM25 search
pub struct HybridRetriever {
    storage: Storage,
    index: IndexManager,
    config: RetrievalConfig,
    raptor_tree: Option<RaptorTree>,
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
        })
    }

    /// Create a new hybrid retriever with the given storage and index
    pub fn new(storage: Storage, index: IndexManager) -> Self {
        Self {
            storage,
            index,
            config: RetrievalConfig::default(),
            raptor_tree: None,
        }
    }

    /// Set the retrieval configuration
    pub fn with_config(mut self, config: RetrievalConfig) -> Self {
        self.config = config;
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

    /// Index a document for retrieval
    pub async fn add_document(&self, doc: &Document) -> Result<()> {
        // Store the document
        let context = self.admin_context("add_document");
        self.storage.store_document(doc, &context).await?;

        // Index in BM25
        self.index.index_document(doc)?;

        // TODO: Generate and store embeddings for vector search

        Ok(())
    }

    /// Delete a document
    pub async fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        let context = self.admin_context("delete_document");
        self.storage.delete_document(doc_id, &context).await?;
        // TODO: Remove from BM25 index as well
        Ok(())
    }

    /// Get retrieval statistics
    pub async fn stats(&self) -> Result<RetrievalStats> {
        let context = self.admin_context("stats");
        let storage_stats = self.storage.stats(&context).await?;
        let index_stats = self.index.stats()?;

        let raptor_stats = if let Some(ref tree) = self.raptor_tree {
            Some(tree.stats())
        } else {
            None
        };

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
                tree.build_from_chunks(&all_chunks, embedder, summarizer).await?;
            }
        }

        Ok(())
    }

    /// Search using hybrid retrieval (vector + BM25)
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        self.search_hybrid(query, None, &RetrievalConfig::default()).await
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
            })
            .collect();

        Ok(results)
    }

    /// Search using hybrid retrieval with custom configuration
    pub async fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        config: &RetrievalConfig,
    ) -> Result<Vec<HybridResult>> {
        let mut all_results = Vec::new();

        // Get sparse results (BM25) if alpha allows
        if config.alpha < 1.0 {
            if let Ok(sparse_results) = self.index.search_bm25(query, config.top_k) {
                let sparse_weight = 1.0 - config.alpha;
                for result in sparse_results {
                    all_results.push(HybridResult {
                        doc_id: result.doc_id,
                        chunk_id: result.chunk_id,
                        text: result.text,
                        score: result.score * sparse_weight,
                        dense_score: None,
                        sparse_score: Some(result.score),
                        match_source: MatchSource::Sparse,
                    });
                }
            }
        }

        // Get dense results (vector search) if alpha allows
        if config.alpha > 0.0 {
            let embedding = if let Some(emb) = query_embedding {
                emb.to_vec()
            } else {
                // TODO: Generate embedding from query
                vec![0.0; 384] // Placeholder
            };

            if let Ok(dense_results) = self.storage.search_by_vector(
                &embedding,
                config.top_k,
                &self.admin_context("search_hybrid"),
            ).await {
                let dense_weight = config.alpha;
                for (chunk_id, score) in dense_results {
                    all_results.push(HybridResult {
                        doc_id: Uuid::nil(), // TODO: Map chunk_id to doc_id
                        chunk_id,
                        text: String::new(), // TODO: Get chunk text
                        score: score * dense_weight,
                        dense_score: Some(score),
                        sparse_score: None,
                        match_source: MatchSource::Dense,
                    });
                }
            }
        }

        // Combine and rerank results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(config.top_k);

        Ok(all_results)
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
    retriever: HybridRetriever,
}

impl KnowledgeBase {
    /// Create a new in-memory knowledge base
    pub fn in_memory() -> Result<Self> {
        Ok(Self {
            retriever: HybridRetriever::in_memory()?,
        })
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
    pub async fn query_with_config(&self, query: &str, config: &RetrievalConfig) -> Result<Vec<HybridResult>> {
        self.retriever.search_hybrid(query, None, config).await
    }

    /// Get statistics
    pub async fn stats(&self) -> Result<RetrievalStats> {
        self.retriever.stats().await
    }

    /// Delete a document
    pub async fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        let context = self.retriever.admin_context("delete_document");
        self.retriever.storage.delete_document(doc_id, &context).await?;
        // TODO: Remove from BM25 index as well
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentType, Source, SourceType, EmbeddingIds, Chunk};
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

        // Query
        let results = kb.query("machine learning", 5).await.unwrap();
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
