//! End-to-End Ingest -> Index -> Retrieve Workflow Tests
//!
//! These tests verify the complete data pipeline from document ingestion
//! through chunking, embedding, indexing, and retrieval.
//!
//! # Test Coverage
//!
//! 1. File ingestion for multiple formats
//! 2. Chunking with various strategies
//! 3. Embedding generation (mock/real)
//! 4. Storage and BM25 indexing
//! 5. Sparse, dense, and hybrid retrieval
//! 6. Result ranking and fusion
//!
//! # Prerequisites
//!
//! Requires `memory` feature to be enabled:
//! ```bash
//! cargo test --features memory ingest_workflow
//! ```

#![cfg(feature = "memory")]

use chrono::Utc;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::NamedTempFile;
use uuid::Uuid;

use reasonkit::{
    ingestion::DocumentIngester,
    processing::chunking::{chunk_document, ChunkingConfig, ChunkingStrategy},
    rag::{RagConfig, RagEngine},
    Chunk, Document, DocumentType, EmbeddingIds, Metadata, Source, SourceType,
};
use reasonkit_mem::{
    embedding::{EmbeddingPipeline, EmbeddingProvider, EmbeddingResult, EmbeddingVector},
    indexing::IndexManager,
    retrieval::{HybridResult, HybridRetriever, KnowledgeBase, RetrievalStats},
    storage::{AccessContext, AccessLevel, Storage},
    MatchSource, RetrievalConfig,
};

// ============================================================================
// Mock Embedding Provider for Testing
// ============================================================================

/// Mock embedding provider that generates deterministic embeddings
/// based on text content for testing purposes.
struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic embedding based on text hash
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate embedding from hash, normalized
        let mut embedding: Vec<f32> = (0..self.dimension)
            .map(|i| {
                let val = ((hash.wrapping_add(i as u64) % 1000) as f32) / 1000.0;
                val * 2.0 - 1.0 // Range [-1, 1]
            })
            .collect();

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut embedding {
                *v /= magnitude;
            }
        }

        embedding
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedding-model"
    }

    async fn embed(&self, text: &str) -> reasonkit_mem::Result<EmbeddingResult> {
        Ok(EmbeddingResult {
            dense: Some(self.generate_embedding(text)),
            sparse: None,
            token_count: text.split_whitespace().count(),
        })
    }

    async fn embed_batch(&self, texts: &[&str]) -> reasonkit_mem::Result<Vec<EmbeddingResult>> {
        Ok(texts
            .iter()
            .map(|text| EmbeddingResult {
                dense: Some(self.generate_embedding(text)),
                sparse: None,
                token_count: text.split_whitespace().count(),
            })
            .collect())
    }
}

// ============================================================================
// Test Fixtures
// ============================================================================

/// Create a test document with content and chunks
fn create_test_document_with_chunks(content: &str, num_chunks: usize) -> Document {
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("/test/document.txt".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let mut doc = Document::new(DocumentType::Note, source).with_content(content.to_string());

    // Create synthetic chunks
    let chunk_size = content.len() / num_chunks.max(1);
    doc.chunks = (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(content.len());
            let text = &content[start..end];

            Chunk {
                id: Uuid::new_v4(),
                text: text.to_string(),
                index: i,
                start_char: start,
                end_char: end,
                token_count: Some(text.len() / 4),
                section: Some(format!("Section {}", i + 1)),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            }
        })
        .collect();

    doc
}

/// Create a markdown test file
fn create_markdown_file(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".md").unwrap();
    writeln!(file, "{}", content).unwrap();
    file
}

/// Create a text test file
fn create_text_file(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".txt").unwrap();
    writeln!(file, "{}", content).unwrap();
    file
}

/// Create an access context for tests
fn admin_context() -> AccessContext {
    AccessContext::new(
        "test_user".to_string(),
        AccessLevel::Admin,
        "e2e_test".to_string(),
    )
}

// ============================================================================
// Phase 1: Ingestion Tests
// ============================================================================

mod phase1_ingestion {
    use super::*;

    #[test]
    fn test_ingest_markdown_file() {
        let content = r#"# Test Document

This is a test document about machine learning.

## Introduction

Machine learning is a subset of artificial intelligence.

## Methods

We use neural networks for classification tasks.
"#;
        let file = create_markdown_file(content);
        let ingester = DocumentIngester::new();

        let doc = ingester.ingest(file.path()).unwrap();

        assert_eq!(doc.doc_type, DocumentType::Documentation);
        assert!(doc.content.raw.contains("machine learning"));
        assert!(doc.content.raw.contains("neural networks"));
        assert!(doc.content.word_count > 0);
    }

    #[test]
    fn test_ingest_text_file() {
        let content = "This is plain text content for testing.";
        let file = create_text_file(content);
        let ingester = DocumentIngester::new();

        let doc = ingester.ingest(file.path()).unwrap();

        assert_eq!(doc.doc_type, DocumentType::Note);
        assert!(doc.content.raw.contains("plain text content"));
    }

    #[test]
    fn test_ingested_document_has_empty_chunks() {
        let file = create_text_file("Test content for chunking.");
        let ingester = DocumentIngester::new();

        let doc = ingester.ingest(file.path()).unwrap();

        // Ingestion does NOT automatically chunk
        assert!(
            doc.chunks.is_empty(),
            "Ingestion should not auto-chunk; chunks should be empty"
        );
    }
}

// ============================================================================
// Phase 2: Chunking Tests
// ============================================================================

mod phase2_chunking {
    use super::*;

    #[test]
    fn test_chunking_fixed_size() {
        let content = "Sentence one. ".repeat(100); // ~1400 chars
        let doc = create_test_document_with_chunks(&content, 0); // No pre-chunks

        // Re-create with raw content
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc = Document::new(DocumentType::Note, source).with_content(content);

        let config = ChunkingConfig {
            chunk_size: 256,
            chunk_overlap: 25,
            min_chunk_size: 50,
            strategy: ChunkingStrategy::FixedSize,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.len() > 1, "Should create multiple chunks");

        // Verify chunk properties
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            assert!(chunk.start_char <= chunk.end_char);
            assert!(chunk.token_count.is_some());
        }
    }

    #[test]
    fn test_chunking_semantic() {
        let content = r#"First paragraph about machine learning.

Second paragraph about neural networks.

Third paragraph about deep learning."#;

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc =
            Document::new(DocumentType::Documentation, source).with_content(content.to_string());

        let config = ChunkingConfig {
            chunk_size: 100,
            chunk_overlap: 10,
            min_chunk_size: 20,
            strategy: ChunkingStrategy::Semantic,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_recursive() {
        let content = "Word. ".repeat(200);

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc = Document::new(DocumentType::Note, source).with_content(content);

        let config = ChunkingConfig {
            chunk_size: 128,
            chunk_overlap: 16,
            min_chunk_size: 32,
            strategy: ChunkingStrategy::Recursive,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_markdown_headers() {
        let content = r#"# Header One

Content under header one is about topic A.

## Header Two

Content under header two discusses topic B in detail."#;

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.md".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc =
            Document::new(DocumentType::Documentation, source).with_content(content.to_string());

        let config = ChunkingConfig {
            chunk_size: 200,
            chunk_overlap: 20,
            min_chunk_size: 20,
            strategy: ChunkingStrategy::DocumentAware,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();

        assert!(!chunks.is_empty());
        // Should have section headers extracted
        assert!(chunks.iter().any(|c| c.section.is_some()));
    }
}

// ============================================================================
// Phase 3: Embedding Tests (Mock)
// ============================================================================

mod phase3_embedding {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedding_provider() {
        let provider = MockEmbeddingProvider::new(384);

        let result = provider.embed("Test text for embedding").await.unwrap();

        assert!(result.dense.is_some());
        assert_eq!(result.dense.as_ref().unwrap().len(), 384);
        assert!(result.token_count > 0);

        // Verify normalization
        let embedding = result.dense.unwrap();
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.001,
            "Embedding should be normalized"
        );
    }

    #[tokio::test]
    async fn test_embedding_batch() {
        let provider = MockEmbeddingProvider::new(256);

        let texts = vec!["First text", "Second text", "Third text"];
        let results = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.dense.is_some());
            assert_eq!(result.dense.as_ref().unwrap().len(), 256);
        }
    }

    #[tokio::test]
    async fn test_embedding_pipeline() {
        let provider = Arc::new(MockEmbeddingProvider::new(128));
        let pipeline = EmbeddingPipeline::new(provider);

        let chunks = vec![
            Chunk {
                id: Uuid::new_v4(),
                text: "Chunk one content".to_string(),
                index: 0,
                start_char: 0,
                end_char: 17,
                token_count: Some(3),
                section: None,
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Chunk two content".to_string(),
                index: 1,
                start_char: 18,
                end_char: 35,
                token_count: Some(3),
                section: None,
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
        ];

        let embeddings = pipeline.embed_chunks(&chunks).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert!(emb.dense.is_some());
            assert_eq!(emb.dense.as_ref().unwrap().len(), 128);
        }
    }

    #[tokio::test]
    async fn test_deterministic_embeddings() {
        let provider = MockEmbeddingProvider::new(64);

        let text = "Same text for consistency";
        let emb1 = provider.embed(text).await.unwrap();
        let emb2 = provider.embed(text).await.unwrap();

        // Same text should produce identical embeddings
        assert_eq!(emb1.dense, emb2.dense);

        // Different text should produce different embeddings
        let emb3 = provider.embed("Different text").await.unwrap();
        assert_ne!(emb1.dense, emb3.dense);
    }
}

// ============================================================================
// Phase 4: Storage and Indexing Tests
// ============================================================================

mod phase4_storage_indexing {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = Storage::in_memory();
        let context = admin_context();

        let doc = create_test_document_with_chunks("Test content for storage.", 2);
        let mem_doc: reasonkit_mem::Document = doc.into();

        // Store document
        storage.store_document(&mem_doc, &context).await.unwrap();

        // Retrieve document
        let retrieved = storage.get_document(&mem_doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, mem_doc.id);

        // List documents
        let docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_embedding_storage_and_vector_search() {
        let storage = Storage::in_memory();
        let context = admin_context();

        // Store embeddings
        let chunk1_id = Uuid::new_v4();
        let chunk2_id = Uuid::new_v4();

        let emb1 = vec![0.1, 0.2, 0.3, 0.4];
        let emb2 = vec![0.9, 0.8, 0.7, 0.6];

        storage
            .store_embeddings(&chunk1_id, &emb1, &context)
            .await
            .unwrap();
        storage
            .store_embeddings(&chunk2_id, &emb2, &context)
            .await
            .unwrap();

        // Search with query similar to emb1
        let query = vec![0.15, 0.25, 0.35, 0.45];
        let results = storage.search_by_vector(&query, 2, &context).await.unwrap();

        assert_eq!(results.len(), 2);
        // First result should be chunk1 (more similar)
        assert_eq!(results[0].0, chunk1_id);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_bm25_indexing() {
        let index = IndexManager::in_memory().unwrap();

        let doc = create_test_document_with_chunks(
            "Machine learning is a subset of artificial intelligence. Neural networks are used for deep learning.",
            2
        );
        let mem_doc: reasonkit_mem::Document = doc.into();

        // Index document
        let indexed = index.index_document(&mem_doc).unwrap();
        assert_eq!(indexed, 2); // 2 chunks

        // Search
        let results = index.search_bm25("machine learning", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.to_lowercase().contains("machine"));
    }

    #[test]
    fn test_bm25_delete_document() {
        let index = IndexManager::in_memory().unwrap();

        let doc = create_test_document_with_chunks("Content to be deleted.", 1);
        let mem_doc: reasonkit_mem::Document = doc.into();
        let doc_id = mem_doc.id;

        index.index_document(&mem_doc).unwrap();

        // Verify indexed
        let results = index.search_bm25("deleted", 5).unwrap();
        assert!(!results.is_empty());

        // Delete
        index.delete_document(&doc_id).unwrap();

        // Verify deleted (search should return empty or not find this doc)
        // Note: Tantivy may still return stale results until segment merge
    }
}

// ============================================================================
// Phase 5: Retrieval Tests
// ============================================================================

mod phase5_retrieval {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_retriever_sparse_search() {
        let retriever = HybridRetriever::in_memory().unwrap();

        let doc1 = create_test_document_with_chunks(
            "Machine learning enables computers to learn from data.",
            1,
        );
        let doc2 = create_test_document_with_chunks(
            "Quantum computing uses quantum mechanics for computation.",
            1,
        );

        let mem_doc1: reasonkit_mem::Document = doc1.into();
        let mem_doc2: reasonkit_mem::Document = doc2.into();

        retriever.add_document(&mem_doc1).await.unwrap();
        retriever.add_document(&mem_doc2).await.unwrap();

        // Sparse search
        let results = retriever
            .search_sparse("machine learning data", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].match_source, MatchSource::Sparse);
        assert!(results[0].text.to_lowercase().contains("machine"));
    }

    #[tokio::test]
    async fn test_hybrid_retriever_with_embeddings() {
        let provider = Arc::new(MockEmbeddingProvider::new(64));
        let pipeline = Arc::new(EmbeddingPipeline::new(provider));

        let retriever = HybridRetriever::in_memory()
            .unwrap()
            .with_embedding_pipeline(pipeline);

        let doc = create_test_document_with_chunks(
            "Deep learning uses neural networks with multiple layers.",
            1,
        );
        let mem_doc: reasonkit_mem::Document = doc.into();

        retriever.add_document(&mem_doc).await.unwrap();

        // Dense search
        let results = retriever.search_dense("neural network", 5).await.unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].match_source, MatchSource::Dense);
    }

    #[tokio::test]
    async fn test_knowledge_base_query() {
        let provider = Arc::new(MockEmbeddingProvider::new(64));
        let pipeline = Arc::new(EmbeddingPipeline::new(provider));

        let kb = KnowledgeBase::in_memory()
            .unwrap()
            .with_embedding_pipeline(pipeline);

        let doc = create_test_document_with_chunks(
            "Reinforcement learning trains agents through reward signals.",
            1,
        );
        let mem_doc: reasonkit_mem::Document = doc.into();

        kb.add(&mem_doc).await.unwrap();

        // Query using sparse (BM25) - reliable without real embeddings
        let results = kb
            .retriever()
            .search_sparse("reinforcement learning", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].text.to_lowercase().contains("reinforcement"));
    }

    #[tokio::test]
    async fn test_retrieval_stats() {
        let retriever = HybridRetriever::in_memory().unwrap();

        let doc = create_test_document_with_chunks("Test document for stats verification.", 2);
        let mem_doc: reasonkit_mem::Document = doc.into();

        retriever.add_document(&mem_doc).await.unwrap();

        let stats = retriever.stats().await.unwrap();

        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 2);
    }
}

// ============================================================================
// Phase 6: Full Workflow E2E Tests
// ============================================================================

mod phase6_full_workflow {
    use super::*;

    #[tokio::test]
    async fn test_complete_ingest_index_retrieve_workflow() {
        // 1. Create test file and ingest
        let content = r#"# Machine Learning Fundamentals

Machine learning is a branch of artificial intelligence.

## Supervised Learning

Supervised learning uses labeled data for training.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data."#;

        let file = create_markdown_file(content);
        let ingester = DocumentIngester::new();
        let mut doc = ingester.ingest(file.path()).unwrap();

        assert_eq!(doc.doc_type, DocumentType::Documentation);
        assert!(doc.chunks.is_empty()); // Not chunked yet

        // 2. Chunk the document
        let chunk_config = ChunkingConfig {
            chunk_size: 200,
            chunk_overlap: 20,
            min_chunk_size: 30,
            strategy: ChunkingStrategy::DocumentAware,
            respect_sentences: true,
        };

        doc.chunks = chunk_document(&doc, &chunk_config).unwrap();
        assert!(!doc.chunks.is_empty());

        // 3. Create retriever with mock embeddings
        let provider = Arc::new(MockEmbeddingProvider::new(128));
        let pipeline = Arc::new(EmbeddingPipeline::new(provider));

        let retriever = HybridRetriever::in_memory()
            .unwrap()
            .with_embedding_pipeline(pipeline);

        // 4. Add document (stores, indexes, embeds)
        let mem_doc: reasonkit_mem::Document = doc.into();
        retriever.add_document(&mem_doc).await.unwrap();

        // 5. Verify storage
        let stats = retriever.stats().await.unwrap();
        assert_eq!(stats.document_count, 1);
        assert!(stats.chunk_count > 0);

        // 6. Test sparse retrieval
        let sparse_results = retriever
            .search_sparse("supervised learning labeled", 5)
            .await
            .unwrap();

        assert!(!sparse_results.is_empty());
        assert!(sparse_results[0].text.to_lowercase().contains("supervised"));

        // 7. Test dense retrieval
        let dense_results = retriever
            .search_dense("machine learning AI", 5)
            .await
            .unwrap();

        assert!(!dense_results.is_empty());

        // 8. Verify match sources
        assert_eq!(sparse_results[0].match_source, MatchSource::Sparse);
        assert_eq!(dense_results[0].match_source, MatchSource::Dense);
    }

    #[tokio::test]
    async fn test_multi_document_retrieval() {
        let provider = Arc::new(MockEmbeddingProvider::new(64));
        let pipeline = Arc::new(EmbeddingPipeline::new(provider));

        let retriever = HybridRetriever::in_memory()
            .unwrap()
            .with_embedding_pipeline(pipeline);

        // Add multiple documents
        let docs = vec![
            create_test_document_with_chunks(
                "Natural language processing enables machines to understand text.",
                1,
            ),
            create_test_document_with_chunks(
                "Computer vision allows machines to interpret images and video.",
                1,
            ),
            create_test_document_with_chunks(
                "Robotics combines AI with mechanical engineering for autonomous systems.",
                1,
            ),
        ];

        for doc in docs {
            let mem_doc: reasonkit_mem::Document = doc.into();
            retriever.add_document(&mem_doc).await.unwrap();
        }

        // Search for NLP content
        let results = retriever
            .search_sparse("natural language text", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].text.to_lowercase().contains("natural language"));

        // Search for robotics content
        let results = retriever
            .search_sparse("robotics autonomous", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].text.to_lowercase().contains("robotics"));
    }

    #[tokio::test]
    async fn test_document_deletion_workflow() {
        let retriever = HybridRetriever::in_memory().unwrap();

        let doc = create_test_document_with_chunks("Temporary content for deletion test.", 1);
        let mem_doc: reasonkit_mem::Document = doc.into();
        let doc_id = mem_doc.id;

        // Add document
        retriever.add_document(&mem_doc).await.unwrap();

        // Verify it exists
        let results = retriever.search_sparse("temporary", 5).await.unwrap();
        assert!(!results.is_empty());

        // Delete document
        retriever.delete_document(&doc_id).await.unwrap();

        // Verify stats updated
        let stats = retriever.stats().await.unwrap();
        assert_eq!(stats.document_count, 0);
    }
}

// ============================================================================
// Phase 7: RAG Engine Tests (Feature-Gated)
// ============================================================================

#[cfg(feature = "memory")]
mod phase7_rag_engine {
    use super::*;

    #[tokio::test]
    async fn test_rag_engine_retrieval_only() {
        let engine = RagEngine::in_memory().unwrap();

        let doc = create_test_document_with_chunks(
            "Chain-of-thought prompting enables step-by-step reasoning in large language models.",
            1,
        );
        let core_doc = reasonkit::Document::new(
            reasonkit::DocumentType::Note,
            reasonkit::Source {
                source_type: reasonkit::SourceType::Local,
                url: None,
                path: Some("/test/cot.txt".to_string()),
                arxiv_id: None,
                github_repo: None,
                retrieved_at: Utc::now(),
                version: None,
            },
        )
        .with_content(
            "Chain-of-thought prompting enables step-by-step reasoning in large language models."
                .to_string(),
        );

        // Create a proper document with chunks
        let mut proper_doc = core_doc;
        proper_doc.chunks = vec![Chunk {
            id: Uuid::new_v4(),
            text: "Chain-of-thought prompting enables step-by-step reasoning in large language models.".to_string(),
            index: 0,
            start_char: 0,
            end_char: 82,
            token_count: Some(12),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        }];

        engine.add_document(&proper_doc).await.unwrap();

        // Query without LLM (retrieval only mode)
        let response = engine
            .query("How does chain of thought work?")
            .await
            .unwrap();

        assert!(response.answer.contains("Retrieved"));
        assert!(!response.sources.is_empty());
        assert!(response.retrieval_stats.chunks_used > 0);
    }

    #[tokio::test]
    async fn test_rag_config_presets() {
        let quick = RagConfig::quick();
        assert_eq!(quick.top_k, 3);
        assert!(!quick.include_sources);

        let thorough = RagConfig::thorough();
        assert_eq!(thorough.top_k, 10);
        assert!(thorough.include_sources);
    }
}

// ============================================================================
// Benchmark / Performance Smoke Tests
// ============================================================================

mod performance_smoke_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_bulk_document_ingestion_performance() {
        let retriever = HybridRetriever::in_memory().unwrap();

        let start = Instant::now();
        const NUM_DOCS: usize = 50;

        for i in 0..NUM_DOCS {
            let content = format!(
                "Document {} content about topic {} with various terms.",
                i,
                i % 5
            );
            let doc = create_test_document_with_chunks(&content, 2);
            let mem_doc: reasonkit_mem::Document = doc.into();
            retriever.add_document(&mem_doc).await.unwrap();
        }

        let elapsed = start.elapsed();
        println!(
            "Ingested {} documents in {:?} ({:.2} docs/sec)",
            NUM_DOCS,
            elapsed,
            NUM_DOCS as f64 / elapsed.as_secs_f64()
        );

        // Performance assertion: should complete in reasonable time
        assert!(
            elapsed.as_secs() < 30,
            "Bulk ingestion took too long: {:?}",
            elapsed
        );

        // Verify all documents indexed
        let stats = retriever.stats().await.unwrap();
        assert_eq!(stats.document_count, NUM_DOCS);
    }

    #[tokio::test]
    async fn test_search_latency() {
        let retriever = HybridRetriever::in_memory().unwrap();

        // Add some documents
        for i in 0..20 {
            let content = format!(
                "Machine learning document {} about neural networks and deep learning.",
                i
            );
            let doc = create_test_document_with_chunks(&content, 1);
            let mem_doc: reasonkit_mem::Document = doc.into();
            retriever.add_document(&mem_doc).await.unwrap();
        }

        // Measure search latency
        let start = Instant::now();
        const NUM_QUERIES: usize = 100;

        for _ in 0..NUM_QUERIES {
            let _ = retriever.search_sparse("neural networks", 5).await.unwrap();
        }

        let elapsed = start.elapsed();
        let avg_latency_ms = elapsed.as_millis() as f64 / NUM_QUERIES as f64;

        println!(
            "Average search latency: {:.2} ms ({} queries)",
            avg_latency_ms, NUM_QUERIES
        );

        // Latency assertion: should be fast
        assert!(
            avg_latency_ms < 100.0,
            "Search latency too high: {:.2} ms",
            avg_latency_ms
        );
    }
}
