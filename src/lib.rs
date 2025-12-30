//! # ReasonKit Core
//!
//! AI Thinking Enhancement System - Turn Prompts into Protocols
//!
//! ## Overview
//!
//! ReasonKit Core is a **PURE REASONING ENGINE** that improves AI thinking patterns
//! through structured reasoning protocols (ThinkTools).
//!
//! ### ThinkTools (THE CORE)
//!
//! - **GigaThink**: Multi-perspective expansion (10+ viewpoints)
//! - **LaserLogic**: Precision deductive reasoning, fallacy detection
//! - **BedRock**: First principles decomposition, axiom rebuilding
//! - **ProofGuard**: Multi-source verification, contradiction detection
//! - **BrutalHonesty**: Adversarial self-critique, find flaws first
//!
//! ### Optional Memory Infrastructure
//!
//! Enable the `memory` feature to add storage, retrieval, and embeddings via `reasonkit-mem`.
//!
//! ## Example
//!
//! ```rust,ignore
//! use reasonkit::thinktool::{ThinkToolExecutor, Profile};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let executor = ThinkToolExecutor::new();
//!     let result = executor.run("Should we use microservices?", Profile::Balanced).await?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```

// TRACKED: Enable `#![warn(missing_docs)]` before v1.0 release
// Status: All public APIs need documentation first (tracked in QA plan)
#![allow(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

// ============================================================================
// CORE MODULES (always available)
pub mod bindings;
pub mod constants;
pub mod error;
pub mod evaluation;
pub mod ingestion;
pub mod m2;
pub mod mcp;
pub mod processing;
pub mod telemetry;
pub mod thinktool;
pub mod verification;
pub mod web;

// Multi-Language Code Intelligence Enhancement
pub mod code_intelligence;

// ============================================================================
// MEMORY MODULES (optional - enable with `memory` feature)
// ============================================================================
// NOTE: Most memory modules (storage, embedding, retrieval, raptor, indexing)
// are provided by reasonkit-mem crate. The rag module remains in core as it
// provides the full RAG engine with LLM integration.
// Re-export reasonkit-mem types when memory feature is enabled
#[cfg(feature = "memory")]
pub use reasonkit_mem;

// Re-export commonly used types from reasonkit-mem for convenience
#[cfg(feature = "memory")]
pub use reasonkit_mem::{
    embedding, indexing, raptor, retrieval, storage, Error as MemError, Result as MemResult,
};

// RAG module remains in core (full engine with LLM integration)
// It uses reasonkit-mem modules for retrieval/storage/indexing
#[cfg(feature = "memory")]
pub mod rag;

#[cfg(feature = "arf")]
pub mod arf;

// Re-exports
pub use error::{Error, Result};

/// Crate version string for runtime logging.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[pymodule]
fn reasonkit(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::Reasoner>()?;
    Ok(())
}

// ============================================================================
// CORE TYPES (always available - needed by ingestion, processing, etc.)
// ============================================================================

/// Document type categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    Paper,
    Documentation,
    Code,
    Note,
    Transcript,
    Benchmark,
}

/// Source type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Arxiv,
    Github,
    Website,
    Local,
    Api,
}

/// Source information for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    #[serde(rename = "type")]
    pub source_type: SourceType,
    pub url: Option<String>,
    pub path: Option<String>,
    pub arxiv_id: Option<String>,
    pub github_repo: Option<String>,
    pub retrieved_at: DateTime<Utc>,
    pub version: Option<String>,
}

/// Author information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    pub name: String,
    pub affiliation: Option<String>,
    pub email: Option<String>,
}

/// Document metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub title: Option<String>,
    pub authors: Vec<Author>,
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
    pub date: Option<String>,
    pub venue: Option<String>,
    pub citations: Option<i32>,
    pub tags: Vec<String>,
    pub categories: Vec<String>,
    pub keywords: Vec<String>,
    pub doi: Option<String>,
    pub license: Option<String>,
}

/// References to different embedding types for a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingIds {
    pub dense: Option<String>,
    pub sparse: Option<String>,
    pub colbert: Option<String>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub token_count: Option<usize>,
    pub section: Option<String>,
    pub page: Option<usize>,
    pub embedding_ids: EmbeddingIds,
}

/// Processing state enumeration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingState {
    #[default]
    Pending,
    Processing,
    Completed,
    Failed,
}

/// Processing status for a document
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub status: ProcessingState,
    pub chunked: bool,
    pub embedded: bool,
    pub indexed: bool,
    pub raptor_processed: bool,
    pub errors: Vec<String>,
}

/// Content format
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentFormat {
    #[default]
    Text,
    Markdown,
    Html,
    Latex,
}

/// Document content
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentContent {
    pub raw: String,
    pub format: ContentFormat,
    pub language: String,
    pub word_count: usize,
    pub char_count: usize,
}

/// A document in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    #[serde(rename = "type")]
    pub doc_type: DocumentType,
    pub source: Source,
    pub content: DocumentContent,
    pub metadata: Metadata,
    pub processing: ProcessingStatus,
    pub chunks: Vec<Chunk>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl Document {
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

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

// Conversion to reasonkit-mem Document type
#[cfg(feature = "memory")]
impl From<Document> for reasonkit_mem::Document {
    fn from(doc: Document) -> Self {
        use reasonkit_mem::types::{
            Author as MemAuthor, Chunk as MemChunk, ContentFormat as MemContentFormat,
            DocumentContent as MemDocumentContent, DocumentType as MemDocumentType,
            EmbeddingIds as MemEmbeddingIds, Metadata as MemMetadata,
            ProcessingState as MemProcessingState, ProcessingStatus as MemProcessingStatus,
            Source as MemSource, SourceType as MemSourceType,
        };

        // Convert DocumentType
        let doc_type = match doc.doc_type {
            DocumentType::Paper => MemDocumentType::Paper,
            DocumentType::Documentation => MemDocumentType::Documentation,
            DocumentType::Code => MemDocumentType::Code,
            DocumentType::Note => MemDocumentType::Note,
            DocumentType::Transcript => MemDocumentType::Transcript,
            DocumentType::Benchmark => MemDocumentType::Benchmark,
        };

        // Convert SourceType
        let source_type = match doc.source.source_type {
            SourceType::Arxiv => MemSourceType::Arxiv,
            SourceType::Github => MemSourceType::Github,
            SourceType::Website => MemSourceType::Website,
            SourceType::Local => MemSourceType::Local,
            SourceType::Api => MemSourceType::Api,
        };

        // Convert Source
        let source = MemSource {
            source_type,
            url: doc.source.url,
            path: doc.source.path,
            arxiv_id: doc.source.arxiv_id,
            github_repo: doc.source.github_repo,
            retrieved_at: doc.source.retrieved_at,
            version: doc.source.version,
        };

        // Convert ContentFormat
        let format = match doc.content.format {
            ContentFormat::Text => MemContentFormat::Text,
            ContentFormat::Markdown => MemContentFormat::Markdown,
            ContentFormat::Html => MemContentFormat::Html,
            ContentFormat::Latex => MemContentFormat::Latex,
        };

        // Convert DocumentContent
        let content = MemDocumentContent {
            raw: doc.content.raw,
            format,
            language: doc.content.language,
            word_count: doc.content.word_count,
            char_count: doc.content.char_count,
        };

        // Convert Authors
        let authors = doc
            .metadata
            .authors
            .into_iter()
            .map(|a| MemAuthor {
                name: a.name,
                affiliation: a.affiliation,
                email: a.email,
            })
            .collect();

        // Convert Metadata
        let metadata = MemMetadata {
            title: doc.metadata.title,
            authors,
            abstract_text: doc.metadata.abstract_text,
            date: doc.metadata.date,
            venue: doc.metadata.venue,
            citations: doc.metadata.citations,
            tags: doc.metadata.tags,
            categories: doc.metadata.categories,
            keywords: doc.metadata.keywords,
            doi: doc.metadata.doi,
            license: doc.metadata.license,
        };

        // Convert ProcessingState
        let status = match doc.processing.status {
            ProcessingState::Pending => MemProcessingState::Pending,
            ProcessingState::Processing => MemProcessingState::Processing,
            ProcessingState::Completed => MemProcessingState::Completed,
            ProcessingState::Failed => MemProcessingState::Failed,
        };

        // Convert ProcessingStatus
        let processing = MemProcessingStatus {
            status,
            chunked: doc.processing.chunked,
            embedded: doc.processing.embedded,
            indexed: doc.processing.indexed,
            raptor_processed: doc.processing.raptor_processed,
            errors: doc.processing.errors,
        };

        // Convert Chunks
        let chunks = doc
            .chunks
            .into_iter()
            .map(|c| {
                let embedding_ids = MemEmbeddingIds {
                    dense: c.embedding_ids.dense,
                    sparse: c.embedding_ids.sparse,
                    colbert: c.embedding_ids.colbert,
                };
                MemChunk {
                    id: c.id,
                    text: c.text,
                    index: c.index,
                    start_char: c.start_char,
                    end_char: c.end_char,
                    token_count: c.token_count,
                    section: c.section,
                    page: c.page,
                    embedding_ids,
                }
            })
            .collect();

        // Construct reasonkit-mem Document
        reasonkit_mem::Document {
            id: doc.id,
            doc_type,
            source,
            content,
            metadata,
            processing,
            chunks,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

/// Source of a search match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    Dense,
    Sparse,
    Hybrid,
    Raptor,
}

/// Search result from a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub document_id: Uuid,
    pub chunk: Chunk,
    pub match_source: MatchSource,
}

// ============================================================================
// MEMORY-SPECIFIC TYPES (only with `memory` feature)
// ============================================================================

#[cfg(feature = "memory")]
pub use reasonkit_mem::RetrievalConfig;

/// Simple retrieval config (available without memory feature)
#[cfg(not(feature = "memory"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub min_score: f32,
    pub alpha: f32,
    pub use_raptor: bool,
    pub rerank: bool,
}

#[cfg(not(feature = "memory"))]
impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.0,
            alpha: 0.7,
            use_raptor: false,
            rerank: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_compiles() {
        // This test verifies basic module compilation
        // The fact that it runs means the crate compiles successfully
    }

    #[test]
    fn test_document_creation() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc = Document::new(DocumentType::Note, source);
        assert_eq!(doc.doc_type, DocumentType::Note);
    }
}
