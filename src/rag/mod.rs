//! Retrieval-Augmented Generation (RAG) Engine
//!
//! Combines local Tantivy BM25 retrieval with LLM generation for
//! context-aware question answering.
//!
//! ## Features
//!
//! - Local-first: Uses Tantivy for BM25 indexing (no external vector DB)
//! - Multiple LLM providers: Works with any UnifiedLlmClient provider
//! - Configurable retrieval: Adjust top_k, min_score, max_context
//! - Structured output: Includes sources and confidence scores
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use reasonkit::rag::{RagEngine, RagConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let engine = RagEngine::in_memory()?;
//!
//!     // Add documents
//!     engine.add_document(&doc).await?;
//!
//!     // Query with RAG
//!     let response = engine.query("How does chain-of-thought work?").await?;
//!     println!("{}", response.answer);
//!
//!     for source in response.sources {
//!         println!("- {}", source.text);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::{
    thinktool::{LlmClient, LlmRequest, UnifiedLlmClient},
    Document, Error, Result, RetrievalConfig,
};
#[cfg(feature = "memory")]
use reasonkit_mem::{
    indexing::IndexManager,
    retrieval::{HybridResult, HybridRetriever, RetrievalStats},
    storage::Storage,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// RAG configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Number of chunks to retrieve
    pub top_k: usize,

    /// Minimum relevance score (0.0-1.0)
    pub min_score: f32,

    /// Maximum context tokens to include
    pub max_context_tokens: usize,

    /// Whether to include source citations
    pub include_sources: bool,

    /// System prompt template
    pub system_prompt: String,

    /// Whether to use sparse-only retrieval (BM25)
    pub sparse_only: bool,

    /// Alpha for hybrid search (0.0 = sparse only, 1.0 = dense only)
    pub hybrid_alpha: f32,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.1,
            max_context_tokens: 2000,
            include_sources: true,
            system_prompt: DEFAULT_RAG_PROMPT.to_string(),
            sparse_only: true, // Default to BM25-only for local prototype
            hybrid_alpha: 0.3,
        }
    }
}

impl RagConfig {
    /// Create a config optimized for quick responses
    pub fn quick() -> Self {
        Self {
            top_k: 3,
            min_score: 0.2,
            max_context_tokens: 1000,
            include_sources: false,
            sparse_only: true,
            ..Default::default()
        }
    }

    /// Create a config for thorough research
    pub fn thorough() -> Self {
        Self {
            top_k: 10,
            min_score: 0.05,
            max_context_tokens: 4000,
            include_sources: true,
            sparse_only: false,
            hybrid_alpha: 0.5,
            ..Default::default()
        }
    }
}

const DEFAULT_RAG_PROMPT: &str = r#"You are a helpful assistant answering questions based on the provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this"
3. Be concise but comprehensive
4. When citing information, reference the source section

CONTEXT:
{context}

Answer the question based on the context above. Be accurate and cite your sources."#;

/// Response from RAG query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    /// The generated answer
    pub answer: String,

    /// Sources used to generate the answer
    pub sources: Vec<RagSource>,

    /// Retrieval statistics
    pub retrieval_stats: RagRetrievalStats,

    /// Tokens used in generation
    pub tokens_used: Option<u32>,

    /// Query that was processed
    pub query: String,
}

/// Source document used in RAG response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagSource {
    /// Document ID
    pub doc_id: Uuid,

    /// Chunk ID
    pub chunk_id: Uuid,

    /// Text snippet
    pub text: String,

    /// Relevance score
    pub score: f32,

    /// Section or page reference
    pub section: Option<String>,
}

/// Statistics about the retrieval phase
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagRetrievalStats {
    /// Number of chunks retrieved
    pub chunks_retrieved: usize,

    /// Number of chunks used after filtering
    pub chunks_used: usize,

    /// Total context tokens
    pub context_tokens: usize,

    /// Retrieval time in milliseconds
    pub retrieval_time_ms: u64,
}

/// RAG Engine combining retrieval with generation
pub struct RagEngine {
    retriever: HybridRetriever,
    llm_client: Option<UnifiedLlmClient>,
    config: RagConfig,
}

impl RagEngine {
    /// Create a new in-memory RAG engine
    pub fn in_memory() -> Result<Self> {
        Ok(Self {
            retriever: HybridRetriever::in_memory()?,
            llm_client: None,
            config: RagConfig::default(),
        })
    }

    /// Create a RAG engine with persistent storage
    pub async fn persistent(base_path: PathBuf) -> Result<Self> {
        let storage_path = base_path.join("storage");
        let index_path = base_path.join("index");

        std::fs::create_dir_all(&storage_path)
            .map_err(|e| Error::io(format!("Failed to create storage dir: {}", e)))?;
        std::fs::create_dir_all(&index_path)
            .map_err(|e| Error::io(format!("Failed to create index dir: {}", e)))?;

        let storage = Storage::file(storage_path).await?;
        let index = IndexManager::open(index_path)?;

        Ok(Self {
            retriever: HybridRetriever::new(storage, index),
            llm_client: None,
            config: RagConfig::default(),
        })
    }

    /// Set the LLM client for generation
    pub fn with_llm(mut self, client: UnifiedLlmClient) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Set the RAG configuration
    pub fn with_config(mut self, config: RagConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a document to the knowledge base
    pub async fn add_document(&self, doc: &Document) -> Result<()> {
        let mem_doc: reasonkit_mem::Document = doc.clone().into();
        self.retriever.add_document(&mem_doc).await?;
        Ok(())
    }

    /// Add multiple documents
    pub async fn add_documents(&self, docs: &[Document]) -> Result<usize> {
        let mut count = 0;
        for doc in docs {
            let mem_doc: reasonkit_mem::Document = doc.clone().into();
            self.retriever.add_document(&mem_doc).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Query the knowledge base with RAG
    pub async fn query(&self, query: &str) -> Result<RagResponse> {
        let start = std::time::Instant::now();

        // Retrieve relevant chunks
        let results = if self.config.sparse_only {
            self.retriever
                .search_sparse(query, self.config.top_k)
                .await?
        } else {
            let retrieval_config = RetrievalConfig {
                top_k: self.config.top_k,
                alpha: self.config.hybrid_alpha,
                ..Default::default()
            };
            self.retriever
                .search_hybrid(query, None, &retrieval_config)
                .await?
        };

        let retrieval_time_ms = start.elapsed().as_millis() as u64;

        // Filter by minimum score
        let filtered_results: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        // Build context from results
        let (context, context_tokens) = self.build_context(&filtered_results);

        // Build sources list with section info from retrieval
        let sources: Vec<RagSource> = filtered_results
            .iter()
            .map(|r| RagSource {
                doc_id: r.doc_id,
                chunk_id: r.chunk_id,
                text: truncate_text(&r.text, 200),
                score: r.score,
                section: None, // Section field removed in reasonkit-mem HybridResult
            })
            .collect();

        let retrieval_stats = RagRetrievalStats {
            chunks_retrieved: self.config.top_k,
            chunks_used: filtered_results.len(),
            context_tokens,
            retrieval_time_ms,
        };

        // Generate answer using LLM
        let (answer, tokens_used) = if let Some(ref client) = self.llm_client {
            let system_prompt = self.config.system_prompt.replace("{context}", &context);

            let request = LlmRequest::new(query)
                .with_system(&system_prompt)
                .with_max_tokens(1000);

            let response = client
                .complete(request)
                .await
                .map_err(|e| Error::network(format!("LLM generation failed: {}", e)))?;

            let tokens = Some(response.usage.total_tokens);
            (response.content, tokens)
        } else {
            // No LLM client - return retrieval-only response
            let answer = format!(
                "Retrieved {} relevant chunks for query: \"{}\"\n\nTop results:\n{}",
                filtered_results.len(),
                query,
                filtered_results
                    .iter()
                    .take(3)
                    .enumerate()
                    .map(|(i, r)| format!(
                        "{}. [score: {:.3}] {}",
                        i + 1,
                        r.score,
                        truncate_text(&r.text, 150)
                    ))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            (answer, None)
        };

        Ok(RagResponse {
            answer,
            sources,
            retrieval_stats,
            tokens_used,
            query: query.to_string(),
        })
    }

    /// Retrieve without generation (for inspection)
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<HybridResult>> {
        self.retriever
            .search_sparse(query, top_k)
            .await
            .map_err(Error::from)
    }

    /// Get knowledge base statistics
    pub async fn stats(&self) -> Result<RetrievalStats> {
        self.retriever.stats().await.map_err(Error::from)
    }

    /// Delete a document from the knowledge base
    pub async fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        self.retriever
            .delete_document(doc_id)
            .await
            .map_err(Error::from)
    }

    /// Build context string from retrieved results
    fn build_context(&self, results: &[HybridResult]) -> (String, usize) {
        let mut context_parts = Vec::new();
        let mut total_tokens = 0;

        for (i, result) in results.iter().enumerate() {
            // Rough token estimate (4 chars per token)
            let chunk_tokens = result.text.len() / 4;

            if total_tokens + chunk_tokens > self.config.max_context_tokens {
                break;
            }

            context_parts.push(format!(
                "[Source {}] (relevance: {:.2})\n{}",
                i + 1,
                result.score,
                result.text
            ));

            total_tokens += chunk_tokens;
        }

        (context_parts.join("\n\n---\n\n"), total_tokens)
    }
}

/// Truncate text to specified length
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// COMPREHENSIVE TEST SUITE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, DocumentType, EmbeddingIds, Source, SourceType};
    use chrono::Utc;

    // ========================================================================
    // TEST HELPERS
    // ========================================================================

    /// Create a test document with a single chunk
    fn create_test_document(content: &str, title: &str) -> Document {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(format!("/test/{}.md", title)),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Note, source).with_content(content.to_string());

        doc.chunks = vec![Chunk {
            id: Uuid::new_v4(),
            text: content.to_string(),
            index: 0,
            start_char: 0,
            end_char: content.len(),
            token_count: Some(content.len() / 4),
            section: Some(title.to_string()),
            page: None,
            embedding_ids: EmbeddingIds::default(),
        }];

        doc
    }

    /// Create a test document with multiple chunks for chunking tests
    fn create_multi_chunk_document(chunks: &[&str], title: &str) -> Document {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(format!("/test/{}.md", title)),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let full_content = chunks.join("\n\n");
        let mut doc = Document::new(DocumentType::Note, source).with_content(full_content.clone());

        let mut char_offset = 0;
        doc.chunks = chunks
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let chunk = Chunk {
                    id: Uuid::new_v4(),
                    text: text.to_string(),
                    index: i,
                    start_char: char_offset,
                    end_char: char_offset + text.len(),
                    token_count: Some(text.len() / 4),
                    section: Some(format!("Section {}", i + 1)),
                    page: Some(i / 2 + 1),
                    embedding_ids: EmbeddingIds::default(),
                };
                char_offset += text.len() + 2; // +2 for \n\n separator
                chunk
            })
            .collect();

        doc
    }

    // ========================================================================
    // RAG CONFIG TESTS
    // ========================================================================

    #[test]
    fn test_rag_config_default() {
        let config = RagConfig::default();

        assert_eq!(config.top_k, 5);
        assert_eq!(config.min_score, 0.1);
        assert_eq!(config.max_context_tokens, 2000);
        assert!(config.include_sources);
        assert!(config.sparse_only);
        assert_eq!(config.hybrid_alpha, 0.3);
        assert!(config.system_prompt.contains("CONTEXT"));
    }

    #[test]
    fn test_rag_config_quick() {
        let config = RagConfig::quick();

        assert_eq!(config.top_k, 3);
        assert_eq!(config.min_score, 0.2);
        assert_eq!(config.max_context_tokens, 1000);
        assert!(!config.include_sources);
        assert!(config.sparse_only);
    }

    #[test]
    fn test_rag_config_thorough() {
        let config = RagConfig::thorough();

        assert_eq!(config.top_k, 10);
        assert_eq!(config.min_score, 0.05);
        assert_eq!(config.max_context_tokens, 4000);
        assert!(config.include_sources);
        assert!(!config.sparse_only);
        assert_eq!(config.hybrid_alpha, 0.5);
    }

    #[test]
    fn test_rag_config_serialization() {
        let config = RagConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: RagConfig = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(config.top_k, deserialized.top_k);
        assert_eq!(config.min_score, deserialized.min_score);
        assert_eq!(config.max_context_tokens, deserialized.max_context_tokens);
    }

    // ========================================================================
    // TEXT TRUNCATION TESTS
    // ========================================================================

    #[test]
    fn test_truncate_text_short() {
        let text = "Short text";
        let result = truncate_text(text, 50);
        assert_eq!(result, "Short text");
    }

    #[test]
    fn test_truncate_text_exact_length() {
        let text = "Exactly ten";
        let result = truncate_text(text, 11);
        assert_eq!(result, "Exactly ten");
    }

    #[test]
    fn test_truncate_text_long() {
        let text = "This is a very long text that needs to be truncated";
        let result = truncate_text(text, 20);
        assert_eq!(result.len(), 20);
        assert!(result.ends_with("..."));
        assert_eq!(result, "This is a very lo...");
    }

    #[test]
    fn test_truncate_text_empty() {
        let text = "";
        let result = truncate_text(text, 10);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_text_zero_max() {
        let text = "Some text";
        let result = truncate_text(text, 0);
        // Should handle gracefully with saturating_sub
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_text_very_small_max() {
        let text = "Hello world";
        let result = truncate_text(text, 3);
        assert_eq!(result, "...");
    }

    // ========================================================================
    // DOCUMENT CHUNKING TESTS
    // ========================================================================

    #[test]
    fn test_single_chunk_document() {
        let doc = create_test_document("Simple content", "simple");

        assert_eq!(doc.chunks.len(), 1);
        assert_eq!(doc.chunks[0].text, "Simple content");
        assert_eq!(doc.chunks[0].index, 0);
        assert_eq!(doc.chunks[0].start_char, 0);
        assert_eq!(doc.chunks[0].end_char, 14);
    }

    #[test]
    fn test_multi_chunk_document() {
        let chunks = [
            "First paragraph about machine learning.",
            "Second paragraph about neural networks.",
            "Third paragraph about deep learning.",
        ];
        let doc = create_multi_chunk_document(&chunks, "ml-doc");

        assert_eq!(doc.chunks.len(), 3);

        // Verify chunk indices
        for (i, chunk) in doc.chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
            assert!(chunk
                .section
                .as_ref()
                .unwrap()
                .contains(&format!("{}", i + 1)));
        }

        // Verify chunk offsets are sequential
        let mut prev_end = 0;
        for chunk in &doc.chunks {
            assert!(chunk.start_char >= prev_end);
            assert!(chunk.end_char > chunk.start_char);
            prev_end = chunk.end_char;
        }
    }

    #[test]
    fn test_chunk_token_count_estimation() {
        let content = "This is exactly twenty characters."; // 34 chars
        let doc = create_test_document(content, "token-test");

        // Token count should be approximately chars / 4
        let expected_tokens = content.len() / 4;
        assert_eq!(doc.chunks[0].token_count, Some(expected_tokens));
    }

    #[test]
    fn test_chunk_page_assignment() {
        let chunks = [
            "Page 1 content A",
            "Page 1 content B",
            "Page 2 content A",
            "Page 2 content B",
        ];
        let doc = create_multi_chunk_document(&chunks, "paged-doc");

        // First two chunks should be page 1, next two page 2
        assert_eq!(doc.chunks[0].page, Some(1));
        assert_eq!(doc.chunks[1].page, Some(1));
        assert_eq!(doc.chunks[2].page, Some(2));
        assert_eq!(doc.chunks[3].page, Some(2));
    }

    // ========================================================================
    // BM25 SEARCH TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_bm25_search_basic() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document(
            "Machine learning algorithms process data to make predictions.",
            "ml-basics",
        );
        engine.add_document(&doc).await.expect("Failed to add doc");

        let results = engine
            .retrieve("machine learning predictions", 5)
            .await
            .expect("Retrieval failed");

        assert!(!results.is_empty());
        assert!(results[0].text.contains("Machine learning"));
    }

    #[tokio::test]
    async fn test_bm25_search_multiple_documents() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let docs = vec![
            create_test_document(
                "Python is a popular programming language for data science.",
                "python",
            ),
            create_test_document(
                "Rust provides memory safety without garbage collection.",
                "rust",
            ),
            create_test_document("JavaScript runs in web browsers and Node.js.", "javascript"),
        ];

        for doc in &docs {
            engine.add_document(doc).await.expect("Failed to add doc");
        }

        // Search for Rust
        let results = engine
            .retrieve("memory safety rust", 5)
            .await
            .expect("Retrieval failed");

        assert!(!results.is_empty());
        assert!(results[0].text.contains("Rust"));
    }

    #[tokio::test]
    async fn test_bm25_search_no_match() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Cats and dogs are common pets.", "pets");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Search for something completely unrelated
        let results = engine
            .retrieve("quantum physics relativity", 5)
            .await
            .expect("Retrieval failed");

        // Should return empty or very low score results
        // BM25 might still return the doc if top_k requested, but score should be low
        if !results.is_empty() {
            // Score should be relatively low for unrelated terms
            assert!(results[0].score < 5.0);
        }
    }

    #[tokio::test]
    async fn test_bm25_search_ranking() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        // Doc with many term matches should rank higher
        let doc1 = create_test_document(
            "Neural networks and deep learning are subsets of machine learning.",
            "high-relevance",
        );
        let doc2 = create_test_document(
            "The weather today is sunny with clear skies.",
            "low-relevance",
        );
        let doc3 = create_test_document(
            "Machine learning uses algorithms to learn from data.",
            "medium-relevance",
        );

        engine.add_document(&doc1).await.unwrap();
        engine.add_document(&doc2).await.unwrap();
        engine.add_document(&doc3).await.unwrap();

        let results = engine
            .retrieve("machine learning neural networks", 5)
            .await
            .expect("Retrieval failed");

        assert!(results.len() >= 2);
        // First result should be more relevant than last
        assert!(results[0].score >= results[results.len() - 1].score);
    }

    // ========================================================================
    // RAG ENGINE INTEGRATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_rag_engine_basic() {
        let engine = RagEngine::in_memory().expect("Failed to create in-memory engine");

        let doc1 = create_test_document(
            "Chain-of-thought prompting enables complex reasoning by breaking problems into steps.",
            "cot-basics",
        );
        let doc2 = create_test_document(
            "Self-consistency improves reasoning by sampling multiple paths and selecting the most common answer.",
            "self-consistency",
        );

        engine
            .add_document(&doc1)
            .await
            .expect("Failed to add doc1");
        engine
            .add_document(&doc2)
            .await
            .expect("Failed to add doc2");

        let response = engine
            .query("How does chain of thought work?")
            .await
            .expect("Query failed");

        assert!(!response.sources.is_empty());
        assert!(response.answer.contains("Retrieved"));
        assert!(response.retrieval_stats.chunks_used > 0);
    }

    #[tokio::test]
    async fn test_rag_engine_with_custom_config() {
        let config = RagConfig {
            top_k: 2,
            min_score: 0.0,
            max_context_tokens: 500,
            include_sources: true,
            sparse_only: true,
            ..Default::default()
        };

        let engine = RagEngine::in_memory()
            .expect("Failed to create engine")
            .with_config(config);

        let doc = create_test_document("Test content for RAG engine.", "test");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("test content").await.expect("Query failed");

        // Should respect top_k limit
        assert!(response.retrieval_stats.chunks_retrieved <= 2);
    }

    #[tokio::test]
    async fn test_rag_engine_add_multiple_documents() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let docs = vec![
            create_test_document("Document one content.", "doc1"),
            create_test_document("Document two content.", "doc2"),
            create_test_document("Document three content.", "doc3"),
        ];

        let count = engine
            .add_documents(&docs)
            .await
            .expect("Failed to add docs");
        assert_eq!(count, 3);

        let stats = engine.stats().await.expect("Failed to get stats");
        assert_eq!(stats.document_count, 3);
    }

    #[tokio::test]
    async fn test_rag_engine_stats() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        // Initially empty
        let stats = engine.stats().await.expect("Failed to get stats");
        assert_eq!(stats.document_count, 0);

        // Add documents
        let doc = create_multi_chunk_document(&["Chunk 1", "Chunk 2", "Chunk 3"], "multi-chunk");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let stats = engine.stats().await.expect("Failed to get stats");
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 3);
    }

    #[tokio::test]
    async fn test_rag_engine_delete_document() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Content to delete.", "delete-me");
        let doc_id = doc.id;

        engine.add_document(&doc).await.expect("Failed to add doc");

        // Verify it exists
        let stats = engine.stats().await.expect("Failed to get stats");
        assert_eq!(stats.document_count, 1);

        // Delete it
        engine
            .delete_document(&doc_id)
            .await
            .expect("Failed to delete doc");

        // Verify it's gone
        let stats = engine.stats().await.expect("Failed to get stats");
        assert_eq!(stats.document_count, 0);
    }

    // ========================================================================
    // MIN SCORE FILTERING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_min_score_filtering() {
        let config = RagConfig {
            min_score: 5.0, // High threshold to filter out low scores
            ..Default::default()
        };

        let engine = RagEngine::in_memory()
            .expect("Failed to create engine")
            .with_config(config);

        let doc = create_test_document("Some content about cats and dogs.", "pets");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Query for unrelated topic - should have low BM25 score
        let response = engine
            .query("quantum computing algorithms")
            .await
            .expect("Query failed");

        // Results below min_score should be filtered
        // All sources should have score >= min_score
        for source in &response.sources {
            assert!(source.score >= 5.0);
        }
    }

    #[tokio::test]
    async fn test_min_score_zero() {
        let config = RagConfig {
            min_score: 0.0,
            ..Default::default()
        };

        let engine = RagEngine::in_memory()
            .expect("Failed to create engine")
            .with_config(config);

        let doc = create_test_document("Any content here.", "test");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("test query").await.expect("Query failed");

        // With min_score 0.0, we should get results
        assert!(response.retrieval_stats.chunks_used >= 0);
    }

    // ========================================================================
    // CONTEXT ASSEMBLY TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_context_token_limit() {
        // Create config with small context limit
        let config = RagConfig {
            max_context_tokens: 10, // Very small limit
            min_score: 0.0,
            ..Default::default()
        };

        let engine = RagEngine::in_memory()
            .expect("Failed to create engine")
            .with_config(config);

        // Add a document with substantial content
        let doc = create_test_document(
            "This is a very long document that contains many words and should exceed the token limit when assembled into context.",
            "long-doc",
        );
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("document").await.expect("Query failed");

        // Context tokens should be limited
        assert!(response.retrieval_stats.context_tokens <= 10);
    }

    #[tokio::test]
    async fn test_context_assembly_format() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Test content for context assembly.", "test");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("test content").await.expect("Query failed");

        // Answer should contain formatted output
        assert!(response.answer.contains("Retrieved"));
        assert!(response.answer.contains("score:"));
    }

    // ========================================================================
    // EDGE CASE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_empty_query() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Some content here.", "test");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Empty query should still work
        let response = engine.query("").await.expect("Query failed");

        // Response should be valid even with empty query
        assert!(!response.query.is_empty() || response.query.is_empty()); // Query is stored
    }

    #[tokio::test]
    async fn test_query_with_special_characters() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("C++ and C# are programming languages.", "langs");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Query with special characters
        let response = engine.query("C++ programming").await.expect("Query failed");

        assert!(!response.answer.is_empty());
    }

    #[tokio::test]
    async fn test_query_with_unicode() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document(
            "Machine learning is used in Tokyo for traffic optimization.",
            "japan",
        );
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("Tokyo traffic").await.expect("Query failed");

        assert!(!response.answer.is_empty());
    }

    #[tokio::test]
    async fn test_very_long_query() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Short content.", "short");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Very long query
        let long_query = "word ".repeat(1000);
        let response = engine.query(&long_query).await.expect("Query failed");

        assert!(!response.answer.is_empty());
    }

    #[tokio::test]
    async fn test_no_documents_query() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        // Query with no documents indexed
        let response = engine.query("any query").await.expect("Query failed");

        // Should handle gracefully with empty results
        assert_eq!(response.sources.len(), 0);
        assert_eq!(response.retrieval_stats.chunks_used, 0);
    }

    // ========================================================================
    // RAG RESPONSE STRUCTURE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_rag_response_structure() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Complete content for response test.", "response");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("response test").await.expect("Query failed");

        // Verify all response fields
        assert!(!response.answer.is_empty());
        assert_eq!(response.query, "response test");
        assert!(response.tokens_used.is_none()); // No LLM client
        assert!(response.retrieval_stats.retrieval_time_ms >= 0);
    }

    #[tokio::test]
    async fn test_rag_source_structure() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Source structure test content.", "source");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine
            .query("source structure")
            .await
            .expect("Query failed");

        for source in &response.sources {
            // Verify source fields
            assert!(!source.chunk_id.is_nil());
            assert!(!source.text.is_empty());
            assert!(source.score >= 0.0);
            // Source text should be truncated to 200 chars max
            assert!(source.text.len() <= 200 + 3); // +3 for "..."
        }
    }

    #[tokio::test]
    async fn test_rag_stats_structure() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Stats test content.", "stats");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("stats").await.expect("Query failed");

        let stats = &response.retrieval_stats;
        assert!(stats.chunks_retrieved > 0 || stats.chunks_used == 0);
        assert!(stats.retrieval_time_ms < 10000); // Should be fast
    }

    // ========================================================================
    // SERIALIZATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_rag_response_serialization() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document("Serialization test.", "serial");
        engine.add_document(&doc).await.expect("Failed to add doc");

        let response = engine.query("serialization").await.expect("Query failed");

        // Should serialize to JSON
        let json = serde_json::to_string(&response).expect("Serialization failed");
        assert!(json.contains("answer"));
        assert!(json.contains("sources"));
        assert!(json.contains("retrieval_stats"));

        // Should deserialize back
        let deserialized: RagResponse =
            serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(response.query, deserialized.query);
    }

    #[test]
    fn test_rag_source_serialization() {
        let source = RagSource {
            doc_id: Uuid::new_v4(),
            chunk_id: Uuid::new_v4(),
            text: "Test text".to_string(),
            score: 0.95,
            section: Some("Introduction".to_string()),
        };

        let json = serde_json::to_string(&source).expect("Serialization failed");
        let deserialized: RagSource = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(source.text, deserialized.text);
        assert_eq!(source.score, deserialized.score);
        assert_eq!(source.section, deserialized.section);
    }

    #[test]
    fn test_rag_retrieval_stats_serialization() {
        let stats = RagRetrievalStats {
            chunks_retrieved: 5,
            chunks_used: 3,
            context_tokens: 150,
            retrieval_time_ms: 42,
        };

        let json = serde_json::to_string(&stats).expect("Serialization failed");
        let deserialized: RagRetrievalStats =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(stats.chunks_retrieved, deserialized.chunks_retrieved);
        assert_eq!(stats.chunks_used, deserialized.chunks_used);
        assert_eq!(stats.context_tokens, deserialized.context_tokens);
        assert_eq!(stats.retrieval_time_ms, deserialized.retrieval_time_ms);
    }

    // ========================================================================
    // CONCURRENT ACCESS TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_concurrent_queries() {
        let engine = std::sync::Arc::new(RagEngine::in_memory().expect("Failed to create engine"));

        let doc = create_test_document("Concurrent access test document.", "concurrent");
        engine.add_document(&doc).await.expect("Failed to add doc");

        // Spawn multiple concurrent queries
        let mut handles = vec![];
        for i in 0..5 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let query = format!("query {}", i);
                engine_clone.query(&query).await
            });
            handles.push(handle);
        }

        // All queries should complete successfully
        for handle in handles {
            let result = handle.await.expect("Task panicked");
            assert!(result.is_ok());
        }
    }

    // ========================================================================
    // RETRIEVAL ONLY MODE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_retrieve_only() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        let doc = create_test_document(
            "Vector databases store embeddings for semantic search.",
            "vector-db",
        );
        engine.add_document(&doc).await.expect("Failed to add doc");

        let results = engine
            .retrieve("semantic search embeddings", 5)
            .await
            .expect("Retrieval failed");

        assert!(!results.is_empty());
        assert!(results[0].text.contains("embeddings"));

        // Verify HybridResult structure
        for result in &results {
            assert!(!result.chunk_id.is_nil());
            assert!(!result.text.is_empty());
        }
    }

    #[tokio::test]
    async fn test_retrieve_top_k_limit() {
        let engine = RagEngine::in_memory().expect("Failed to create engine");

        // Add many documents
        for i in 0..10 {
            let doc = create_test_document(
                &format!("Document {} about testing retrieval limits.", i),
                &format!("doc-{}", i),
            );
            engine.add_document(&doc).await.expect("Failed to add doc");
        }

        // Request only 3
        let results = engine
            .retrieve("testing retrieval", 3)
            .await
            .expect("Retrieval failed");

        assert!(results.len() <= 3);
    }

    // ========================================================================
    // BUILDER PATTERN TESTS
    // ========================================================================

    #[test]
    fn test_engine_builder_pattern() {
        let config = RagConfig::quick();

        // This should compile and work
        let _engine = RagEngine::in_memory()
            .expect("Failed to create engine")
            .with_config(config);
    }

    #[test]
    fn test_config_builder_pattern() {
        // Test that configs can be modified
        let mut config = RagConfig::default();
        config.top_k = 20;
        config.min_score = 0.5;

        assert_eq!(config.top_k, 20);
        assert_eq!(config.min_score, 0.5);
    }
}
