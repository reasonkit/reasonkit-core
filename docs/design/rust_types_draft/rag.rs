//! RAG Pipeline Type Definitions
//!
//! Types for the 5-layer RAG (Retrieval-Augmented Generation) pipeline.
//! Reference: docs/design/RAG_PIPELINE_ARCHITECTURE.md

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::types::common::{Confidence, Id, Metadata, Source, TokenUsage};

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENT TYPES (LAYER 1: INGESTION)
// ═══════════════════════════════════════════════════════════════════════════

/// Unique document identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    pub fn new() -> Self {
        Self(Id::new("doc").to_string())
    }

    pub fn from_hash(content_hash: &str) -> Self {
        Self(format!("doc_{}", &content_hash[..12]))
    }
}

impl Default for DocumentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Document format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    Pdf,
    Markdown,
    Html,
    Text,
    Json,
    Jsonl,
    Csv,
    Docx,
    Rst,
    Latex,
    Unknown,
}

impl DocumentFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "pdf" => Self::Pdf,
            "md" | "markdown" => Self::Markdown,
            "html" | "htm" => Self::Html,
            "txt" => Self::Text,
            "json" => Self::Json,
            "jsonl" | "ndjson" => Self::Jsonl,
            "csv" => Self::Csv,
            "docx" => Self::Docx,
            "rst" => Self::Rst,
            "tex" | "latex" => Self::Latex,
            _ => Self::Unknown,
        }
    }

    /// Get MIME type
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Pdf => "application/pdf",
            Self::Markdown => "text/markdown",
            Self::Html => "text/html",
            Self::Text => "text/plain",
            Self::Json => "application/json",
            Self::Jsonl => "application/x-ndjson",
            Self::Csv => "text/csv",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Rst => "text/x-rst",
            Self::Latex => "application/x-latex",
            Self::Unknown => "application/octet-stream",
        }
    }
}

/// Document category for organization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentCategory {
    /// Academic research paper
    AcademicPaper,
    /// Technical documentation
    Documentation,
    /// Code file
    Code,
    /// Configuration file
    Config,
    /// API specification
    ApiSpec,
    /// Tutorial or guide
    Tutorial,
    /// Blog post or article
    Article,
    /// Book or book chapter
    Book,
    /// Other/unclassified
    Other,
}

/// A document in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub title: String,
    pub content: String,
    pub format: DocumentFormat,
    pub category: DocumentCategory,
    pub source: Source,
    pub content_hash: String,
    pub token_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Metadata,
    pub chunks: Vec<ChunkId>,
}

impl Document {
    /// Create a new document
    pub fn new(
        title: impl Into<String>,
        content: impl Into<String>,
        format: DocumentFormat,
        source: Source,
    ) -> Self {
        let content = content.into();
        let content_hash = Self::compute_hash(&content);
        let id = DocumentId::from_hash(&content_hash);
        let now = Utc::now();

        Self {
            id,
            title: title.into(),
            token_count: Self::estimate_tokens(&content),
            content,
            format,
            category: DocumentCategory::Other,
            source,
            content_hash,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            chunks: Vec::new(),
        }
    }

    /// Compute content hash
    fn compute_hash(content: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Estimate token count (rough approximation)
    fn estimate_tokens(content: &str) -> usize {
        // Rough estimate: ~4 characters per token for English
        content.len() / 4
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CHUNK TYPES (LAYER 2: PROCESSING)
// ═══════════════════════════════════════════════════════════════════════════

/// Unique chunk identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn new(doc_id: &DocumentId, index: usize) -> Self {
        Self(format!("{}_chunk_{:04}", doc_id.0, index))
    }
}

/// Chunk type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    /// Regular text chunk
    Text,
    /// Code block
    Code,
    /// Table data
    Table,
    /// Section heading
    Heading,
    /// List or enumeration
    List,
    /// Image caption/description
    Image,
    /// Mathematical formula
    Math,
    /// RAPTOR summary node
    RaptorSummary,
}

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    pub chunk_type: ChunkType,
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub token_count: usize,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}

/// Chunk-specific metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Section hierarchy (e.g., ["Chapter 1", "Section 1.2"])
    pub section_path: Vec<String>,
    /// Heading level (1-6)
    pub heading_level: Option<u8>,
    /// Previous chunk ID for context
    pub prev_chunk: Option<ChunkId>,
    /// Next chunk ID for context
    pub next_chunk: Option<ChunkId>,
    /// RAPTOR tree level (0 = leaf)
    pub raptor_level: u8,
    /// Parent summary chunk for RAPTOR
    pub raptor_parent: Option<ChunkId>,
    /// Child chunks for RAPTOR
    pub raptor_children: Vec<ChunkId>,
    /// Additional key-value metadata
    pub custom: HashMap<String, String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDING TYPES (LAYER 3: EMBEDDING)
// ═══════════════════════════════════════════════════════════════════════════

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub name: String,
    pub provider: EmbeddingProvider,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub normalize: bool,
}

/// Supported embedding providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    OpenAI,
    Cohere,
    Voyage,
    HuggingFace,
    Local,
}

/// Batch embedding request
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub texts: Vec<String>,
    pub model: String,
    pub dimensions: Option<usize>,
}

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimensions: usize,
    pub usage: TokenUsage,
}

// ═══════════════════════════════════════════════════════════════════════════
// INDEX TYPES (LAYER 4: INDEXING)
// ═══════════════════════════════════════════════════════════════════════════

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub index_type: IndexType,
    pub dimensions: Option<usize>,
    pub distance_metric: DistanceMetric,
    pub hnsw_config: Option<HnswConfig>,
    pub bm25_config: Option<Bm25Config>,
}

/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// BM25 full-text index (Tantivy)
    Bm25,
    /// HNSW vector index (Qdrant)
    Hnsw,
    /// Combined hybrid index
    Hybrid,
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// HNSW index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of neighbors per node
    pub m: usize,
    /// Construction-time expansion factor
    pub ef_construct: usize,
    /// Search-time expansion factor
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construct: 100,
            ef_search: 50,
        }
    }
}

/// BM25 index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    /// BM25 k1 parameter
    pub k1: f64,
    /// BM25 b parameter
    pub b: f64,
    /// Stemming language
    pub language: String,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            language: "en".to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RETRIEVAL TYPES (LAYER 5: RETRIEVAL)
// ═══════════════════════════════════════════════════════════════════════════

/// Search/retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum relevance threshold
    pub threshold: f64,
    /// Search method
    pub method: SearchMethod,
    /// Hybrid search alpha (0=BM25, 1=vector)
    pub alpha: f64,
    /// Fusion method for hybrid
    pub fusion: FusionMethod,
    /// RRF k parameter
    pub rrf_k: u32,
    /// Enable RAPTOR hierarchical retrieval
    pub use_raptor: bool,
    /// RAPTOR traversal depth
    pub raptor_depth: usize,
    /// Enable query expansion
    pub expand_query: bool,
    /// Enable reranking
    pub rerank: bool,
    /// Reranking model
    pub rerank_model: Option<String>,
    /// Source filters
    pub source_filter: Option<Vec<String>>,
    /// Document type filters
    pub doc_type_filter: Option<Vec<DocumentCategory>>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            threshold: 0.5,
            method: SearchMethod::Hybrid,
            alpha: 0.5,
            fusion: FusionMethod::Rrf,
            rrf_k: 60,
            use_raptor: false,
            raptor_depth: 3,
            expand_query: false,
            rerank: false,
            rerank_model: None,
            source_filter: None,
            doc_type_filter: None,
        }
    }
}

/// Search methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMethod {
    Bm25,
    Vector,
    Hybrid,
}

/// Fusion methods for hybrid search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion
    Rrf,
    /// Linear combination
    Linear,
    /// Weighted average
    Weighted,
}

/// A search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    pub score: f64,
    pub rank: usize,
    pub source: ResultSource,
    pub metadata: SearchResultMetadata,
}

/// Source of the search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSource {
    pub document_title: String,
    pub section_path: Vec<String>,
    pub source_name: String,
    pub source_url: Option<String>,
    pub page_number: Option<u32>,
}

/// Search result metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// BM25 score component
    pub bm25_score: Option<f64>,
    /// Vector similarity score component
    pub vector_score: Option<f64>,
    /// Reranker score (if applied)
    pub rerank_score: Option<f64>,
    /// RAPTOR level (0 = leaf, higher = summary)
    pub raptor_level: u8,
    /// Confidence in this result
    pub confidence: Confidence,
}

/// Complete retrieval response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub total_candidates: usize,
    pub method: SearchMethod,
    pub execution_time_ms: u64,
    pub token_usage: TokenUsage,
}

// ═══════════════════════════════════════════════════════════════════════════
// RAPTOR TREE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// RAPTOR tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorNode {
    pub id: ChunkId,
    pub level: u8,
    pub content: String,
    pub embedding: Vec<f32>,
    pub children: Vec<ChunkId>,
    pub parent: Option<ChunkId>,
    pub source_documents: Vec<DocumentId>,
}

/// RAPTOR tree configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorConfig {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Chunks per cluster for summarization
    pub cluster_size: usize,
    /// Summarization model
    pub summarizer_model: String,
    /// Maximum summary tokens
    pub max_summary_tokens: usize,
}

impl Default for RaptorConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            cluster_size: 10,
            summarizer_model: "claude-haiku-3.6".to_string(),
            max_summary_tokens: 512,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Pipeline stage for processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Ingestion,
    Processing,
    Embedding,
    Indexing,
    Retrieval,
}

/// Processing status for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub document_id: DocumentId,
    pub stage: PipelineStage,
    pub progress: f64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

/// Pipeline statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    pub documents_processed: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub total_tokens: usize,
    pub total_cost_usd: f64,
    pub avg_chunk_size: f64,
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_format_detection() {
        assert_eq!(DocumentFormat::from_extension("pdf"), DocumentFormat::Pdf);
        assert_eq!(DocumentFormat::from_extension("MD"), DocumentFormat::Markdown);
        assert_eq!(DocumentFormat::from_extension("unknown"), DocumentFormat::Unknown);
    }

    #[test]
    fn test_document_creation() {
        let doc = Document::new(
            "Test Document",
            "This is test content.",
            DocumentFormat::Text,
            Source {
                name: "test".to_string(),
                path: None,
                url: None,
                version: None,
                retrieved_at: None,
            },
        );
        assert!(doc.id.0.starts_with("doc_"));
        assert!(doc.token_count > 0);
    }

    #[test]
    fn test_chunk_id_generation() {
        let doc_id = DocumentId("doc_abc123".to_string());
        let chunk_id = ChunkId::new(&doc_id, 5);
        assert_eq!(chunk_id.0, "doc_abc123_chunk_0005");
    }

    #[test]
    fn test_retrieval_config_defaults() {
        let config = RetrievalConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.method, SearchMethod::Hybrid);
        assert_eq!(config.rrf_k, 60);
    }
}
