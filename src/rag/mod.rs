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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, DocumentType, EmbeddingIds, Source, SourceType};
    use chrono::Utc;

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

    #[tokio::test]
    async fn test_rag_engine_basic() {
        let engine = RagEngine::in_memory().expect("Failed to create in-memory engine");

        // Add test documents
        let doc1 = create_test_document(
            "Chain-of-thought prompting enables complex reasoning by breaking problems into steps.",
            "cot-basics",
        );
        let doc2 = create_test_document(
            "Self-consistency improves reasoning by sampling multiple paths and selecting the most common answer.",
            "self-consistency"
        );

        engine
            .add_document(&doc1)
            .await
            .expect("Failed to add doc1");
        engine
            .add_document(&doc2)
            .await
            .expect("Failed to add doc2");

        // Query (without LLM - retrieval only mode)
        let response = engine
            .query("How does chain of thought work?")
            .await
            .expect("Query failed");

        assert!(!response.sources.is_empty());
        assert!(response.answer.contains("Retrieved"));
        assert!(response.retrieval_stats.chunks_used > 0);
    }

    #[tokio::test]
    async fn test_rag_retrieve_only() {
        let engine = RagEngine::in_memory().expect("Failed to create in-memory engine");

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
    }

    #[tokio::test]
    async fn test_rag_config_quick() {
        let config = RagConfig::quick();

        assert_eq!(config.top_k, 3);
        assert!(config.sparse_only);
        assert!(!config.include_sources);
    }

    #[tokio::test]
    async fn test_rag_config_thorough() {
        let config = RagConfig::thorough();

        assert_eq!(config.top_k, 10);
        assert!(!config.sparse_only);
        assert!(config.include_sources);
    }
}
