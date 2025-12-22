//! # ReasonKit Core
//!
//! A Rust-first knowledge base and RAG system for AI reasoning enhancement.
//!
//! ## Overview
//!
//! ReasonKit Core provides:
//! - **Document Ingestion**: PDF, Markdown, HTML, JSON processing
//! - **Embedding**: Dense, sparse, and ColBERT-style embeddings
//! - **Indexing**: HNSW + BM25 hybrid search
//! - **Retrieval**: RAPTOR-style hierarchical retrieval
//! - **Storage**: Qdrant vector database integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ LAYER 5: RETRIEVAL & QUERY                                  │
//! │   Hybrid Search | RAPTOR Tree Query | Reranking             │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 4: INDEXING                                           │
//! │   HNSW Index | BM25 Index | RAPTOR Tree                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 3: EMBEDDING                                          │
//! │   Dense Embed | Sparse Embed | ColBERT                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 2: PROCESSING                                         │
//! │   Chunking | Cleaning | Metadata Extraction                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 1: INGESTION                                          │
//! │   PDF | HTML/MD | JSON/JSONL | GitHub                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use reasonkit_core::{KnowledgeBase, Document, RetrievalConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize knowledge base
//!     let kb = KnowledgeBase::new("./data").await?;
//!
//!     // Ingest a document
//!     let doc = Document::from_pdf("paper.pdf").await?;
//!     kb.ingest(doc).await?;
//!
//!     // Query
//!     let results = kb.query("What is chain-of-thought prompting?", 5).await?;
//!
//!     for result in results {
//!         println!("Score: {:.3} - {}", result.score, result.chunk.text);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod error;
pub mod ingestion;
pub mod processing;
pub mod embedding;
pub mod indexing;
pub mod retrieval;
pub mod storage;
pub mod thinktool;
pub mod raptor;

#[cfg(feature = "arf")]
pub mod arf;

// Re-exports
pub use error::{Error, Result};

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Document type categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    /// Academic paper (PDF)
    Paper,
    /// Technical documentation
    Documentation,
    /// Source code
    Code,
    /// User notes
    Note,
    /// Meeting/interview transcript
    Transcript,
    /// Benchmark data
    Benchmark,
}

/// Source information for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source type
    #[serde(rename = "type")]
    pub source_type: SourceType,
    /// Original URL
    pub url: Option<String>,
    /// Local file path
    pub path: Option<String>,
    /// arXiv ID (e.g., "2401.18059")
    pub arxiv_id: Option<String>,
    /// GitHub repository (e.g., "anthropics/claude-code")
    pub github_repo: Option<String>,
    /// When the document was retrieved
    pub retrieved_at: DateTime<Utc>,
    /// Version or commit hash
    pub version: Option<String>,
}

/// Source type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// arXiv paper
    Arxiv,
    /// GitHub repository
    Github,
    /// Website
    Website,
    /// Local file
    Local,
    /// API response
    Api,
}

/// Document metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Document title
    pub title: Option<String>,
    /// Authors
    pub authors: Vec<Author>,
    /// Abstract or summary
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
    /// Publication/creation date
    pub date: Option<String>,
    /// Publication venue
    pub venue: Option<String>,
    /// Citation count
    pub citations: Option<i32>,
    /// User-defined tags
    pub tags: Vec<String>,
    /// ReasonKit categories
    pub categories: Vec<String>,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Digital Object Identifier
    pub doi: Option<String>,
    /// Content license
    pub license: Option<String>,
}

/// Author information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// Author name
    pub name: String,
    /// Affiliation
    pub affiliation: Option<String>,
    /// Email
    pub email: Option<String>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: Uuid,
    /// Chunk text content
    pub text: String,
    /// Position in document
    pub index: usize,
    /// Start character offset
    pub start_char: usize,
    /// End character offset
    pub end_char: usize,
    /// Token count (approximate)
    pub token_count: Option<usize>,
    /// Section heading
    pub section: Option<String>,
    /// Page number (for PDFs)
    pub page: Option<usize>,
    /// Associated embedding IDs
    pub embedding_ids: EmbeddingIds,
}

/// References to different embedding types for a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingIds {
    /// Dense embedding ID (in Qdrant)
    pub dense: Option<String>,
    /// Sparse embedding ID
    pub sparse: Option<String>,
    /// ColBERT embedding ID
    pub colbert: Option<String>,
}

/// Processing status for a document
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatus {
    /// Overall status
    pub status: ProcessingState,
    /// Has been chunked
    pub chunked: bool,
    /// Has been embedded
    pub embedded: bool,
    /// Has been indexed
    pub indexed: bool,
    /// Has RAPTOR tree been built
    pub raptor_processed: bool,
    /// Error messages
    pub errors: Vec<String>,
}

/// Processing state enumeration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingState {
    /// Not yet processed
    #[default]
    Pending,
    /// Currently processing
    Processing,
    /// Successfully completed
    Completed,
    /// Failed with errors
    Failed,
}

/// A document in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: Uuid,
    /// Document type
    #[serde(rename = "type")]
    pub doc_type: DocumentType,
    /// Source information
    pub source: Source,
    /// Raw content
    pub content: DocumentContent,
    /// Metadata
    pub metadata: Metadata,
    /// Processing status
    pub processing: ProcessingStatus,
    /// Document chunks
    pub chunks: Vec<Chunk>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: Option<DateTime<Utc>>,
}

/// Document content
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentContent {
    /// Full extracted text
    pub raw: String,
    /// Content format
    pub format: ContentFormat,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
}

/// Content format
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentFormat {
    /// Plain text
    #[default]
    Text,
    /// Markdown
    Markdown,
    /// HTML
    Html,
    /// LaTeX
    Latex,
}

impl Document {
    /// Create a new document with the given type and source
    pub fn new(doc_type: DocumentType, source: Source) -> Self {
        Self {
            id: Uuid::new_v4(),
            doc_type,
            source,
            content: DocumentContent::default(),
            metadata: Metadata::default(),
            processing: ProcessingStatus::default(),
            chunks: Vec::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Set the raw content
    pub fn with_content(mut self, raw: String) -> Self {
        let word_count = raw.split_whitespace().count();
        let char_count = raw.len();
        self.content = DocumentContent {
            raw,
            format: ContentFormat::Text,
            language: "en".to_string(),
            word_count,
            char_count,
        };
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Search result from a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Relevance score
    pub score: f32,
    /// Document ID
    pub document_id: Uuid,
    /// Chunk that matched
    pub chunk: Chunk,
    /// Source of the match (dense, sparse, hybrid)
    pub match_source: MatchSource,
}

/// Source of a search match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    /// Dense vector match
    Dense,
    /// Sparse (BM25) match
    Sparse,
    /// Hybrid search
    Hybrid,
    /// RAPTOR tree match
    Raptor,
}

/// Configuration for retrieval operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum score threshold
    pub min_score: f32,
    /// Alpha for hybrid search (0 = sparse only, 1 = dense only)
    pub alpha: f32,
    /// Whether to use RAPTOR tree
    pub use_raptor: bool,
    /// Whether to rerank results
    pub rerank: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.0,
            alpha: 0.7,  // Favor semantic search
            use_raptor: false,
            rerank: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let source = Source {
            source_type: SourceType::Arxiv,
            url: Some("https://arxiv.org/abs/2401.18059".to_string()),
            path: None,
            arxiv_id: Some("2401.18059".to_string()),
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Paper, source)
            .with_content("This is a test paper about RAPTOR.".to_string());

        assert_eq!(doc.doc_type, DocumentType::Paper);
        assert_eq!(doc.content.word_count, 7);
        assert!(doc.content.raw.contains("RAPTOR"));
    }

    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.alpha, 0.7);
        assert!(!config.use_raptor);
    }
}
