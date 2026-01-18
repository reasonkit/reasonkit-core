#![doc = include_str!("../README.md")]
// doc_auto_cfg was merged into doc_cfg in Rust 1.92
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # ReasonKit Core
//!
//! AI Thinking Enhancement System - Turn Prompts into Protocols
//!
//! ReasonKit Core is a **pure reasoning engine** that improves AI thinking patterns
//! through structured reasoning protocols called ThinkTools. It transforms ad-hoc
//! LLM prompting into auditable, reproducible reasoning chains.
//!
//! ## Philosophy
//!
//! **"Designed, Not Dreamed"** - Structure beats raw intelligence. By imposing
//! systematic reasoning protocols, ReasonKit helps AI models produce more reliable,
//! verifiable, and explainable outputs.
//!
//! ## Quick Start
//!
//! ### Rust Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create executor (auto-detects LLM from environment)
//!     let executor = ProtocolExecutor::new()?;
//!
//!     // Run GigaThink for multi-perspective analysis
//!     let result = executor.execute(
//!         "gigathink",
//!         ProtocolInput::query("Should we use microservices?")
//!     ).await?;
//!
//!     println!("Confidence: {:.2}", result.confidence);
//!     for perspective in result.perspectives() {
//!         println!("- {}", perspective);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Python Usage
//!
//! ```python
//! from reasonkit import Reasoner, Profile, run_gigathink
//!
//! # Quick usage with convenience functions
//! result = run_gigathink("What factors drive startup success?")
//! print(result.perspectives)
//!
//! # Full control with Reasoner class
//! r = Reasoner()
//! result = r.think_with_profile(Profile.Balanced, "Should we pivot?")
//! print(f"Confidence: {result.confidence:.1%}")
//! ```
//!
//! ## ThinkTools (Core Reasoning Protocols)
//!
//! ReasonKit provides five core ThinkTools, each implementing a specific reasoning strategy:
//!
//! | Tool | Code | Purpose | Output |
//! |------|------|---------|--------|
//! | **GigaThink** | `gt` | Expansive creative thinking | 10+ diverse perspectives |
//! | **LaserLogic** | `ll` | Precision deductive reasoning | Validity assessment, fallacy detection |
//! | **BedRock** | `br` | First principles decomposition | Core axioms, rebuilt foundations |
//! | **ProofGuard** | `pg` | Multi-source verification | Triangulated evidence (3+ sources) |
//! | **BrutalHonesty** | `bh` | Adversarial self-critique | Flaws, weaknesses, counter-arguments |
//!
//! ## Reasoning Profiles
//!
//! Profiles chain multiple ThinkTools together for comprehensive analysis:
//!
//! | Profile | ThinkTools | Min Confidence | Use Case |
//! |---------|------------|----------------|----------|
//! | `quick` | GT, LL | 70% | Fast initial analysis |
//! | `balanced` | GT, LL, BR, PG | 80% | Standard decision-making |
//! | `deep` | All 5 | 85% | Complex problems |
//! | `paranoid` | All 5 + validation | 95% | High-stakes decisions |
//!
//! ## Feature Flags
//!
//! - `memory` - Enable memory layer integration via `reasonkit-mem`
//! - `aesthetic` - Enable UI/UX assessment capabilities
//! - `vibe` - Enable VIBE protocol validation system
//! - `code-intelligence` - Enable multi-language code analysis
//! - `arf` - Enable Autonomous Reasoning Framework
//! - `minimax` - Enable MiniMax M2 model integration
//!
//! ## Supported LLM Providers
//!
//! ReasonKit supports 18+ LLM providers out of the box:
//!
//! - **Major Cloud**: Anthropic, OpenAI, Google Gemini, Vertex AI, Azure OpenAI, AWS Bedrock
//! - **Specialized**: xAI (Grok), Groq, Mistral, DeepSeek, Cohere, Perplexity, Cerebras
//! - **Inference**: Together AI, Fireworks AI, Alibaba Qwen
//! - **Aggregation**: OpenRouter (300+ models), Cloudflare AI Gateway
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   User Query     | --> | Protocol Engine  | --> |  Auditable Output|
//! +------------------+     +------------------+     +------------------+
//!                                  |
//!                    +-------------+-------------+
//!                    |             |             |
//!               +----v----+  +-----v-----+  +----v----+
//!               | LLM     |  | ThinkTool |  | Profile |
//!               | Client  |  | Modules   |  | System  |
//!               +---------+  +-----------+  +---------+
//! ```
//!
//! ## Modules
//!
//! - [`thinktool`] - Core ThinkTool protocols and execution engine
//! - [`engine`] - High-level async reasoning loop with streaming
//! - [`orchestration`] - Long-horizon task orchestration (100+ tool calls)
//! - [`error`] - Error types and result aliases
//! - [`telemetry`] - Metrics and observability
//!
//! ## Optional Modules (Feature-Gated)
//!
//! - \[`bindings`\] - Python bindings via PyO3 (requires `python`)
//! - \[`rag`\] - Full RAG engine with LLM integration (requires `memory`)
//! - \[`aesthetic`\] - UI/UX assessment system (requires `aesthetic`)
//! - \[`vibe`\] - VIBE protocol validation (requires `vibe`)
//! - \[`code_intelligence`\] - Multi-language code analysis (requires `code-intelligence`)

// TRACKED: Enable `#![warn(missing_docs)]` before v1.0 release
// Status: All public APIs need documentation first (tracked in QA plan)
#![allow(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

// ============================================================================
// CORE MODULES (always available)
// ============================================================================

/// Python bindings via PyO3 for using ReasonKit from Python.
///
/// Build with `maturin develop --release` for development or
/// `maturin build --release` for distribution.
///
/// See module documentation for Python usage examples.
#[cfg(feature = "python")]
pub mod bindings;

/// Global constants and configuration defaults.
pub mod constants;

/// High-performance async reasoning engine with streaming support.
///
/// The engine module provides [`ReasoningLoop`](engine::ReasoningLoop) for
/// orchestrating ThinkTool execution with memory integration and concurrent
/// processing.
pub mod engine;

/// Error types and result aliases for ReasonKit operations.
///
/// All ReasonKit functions return [`Result<T>`](Result) which is an alias
/// for `std::result::Result<T, Error>`.
pub mod error;

/// Evaluation and benchmarking utilities.
pub mod evaluation;

/// Provider-neutral LLM clients (e.g. Ollama `/api/chat`).
pub mod llm;

/// Document ingestion and processing pipeline.
pub mod ingestion;

/// MiniMax M2 model integration for 100+ tool calling.
///
/// Provides protocol generation, benchmarking, and long-horizon execution
/// capabilities leveraging M2's exceptional tool-use performance.
pub mod m2;

/// MCP (Model Context Protocol) server implementations.
///
/// ReasonKit implements MCP servers in Rust (no Node.js) for tool integration.
pub mod mcp;

/// Long-horizon task orchestration system.
///
/// Coordinates complex multi-step operations across ReasonKit components
/// with state persistence, error recovery, and performance monitoring.
pub mod orchestration;

/// Document processing and transformation utilities.
pub mod processing;

/// Telemetry, metrics, and observability infrastructure.
///
/// Provides OpenTelemetry integration for tracing, metrics collection,
/// and privacy-preserving data export.
pub mod telemetry;

/// ThinkTool protocol engine - the core of ReasonKit.
///
/// This module provides the structured reasoning protocols that transform
/// ad-hoc LLM prompting into auditable, reproducible reasoning chains.
///
/// # Key Types
///
/// - [`ProtocolExecutor`](thinktool::ProtocolExecutor) - Executes protocols with LLM integration
/// - [`ProtocolInput`](thinktool::ProtocolInput) - Input data for protocol execution
/// - [`ProtocolOutput`](thinktool::ProtocolOutput) - Results with confidence scores
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};
///
/// let executor = ProtocolExecutor::new()?;
/// let result = executor.execute(
///     "gigathink",
///     ProtocolInput::query("Analyze market trends")
/// ).await?;
/// ```
pub mod thinktool;

/// Verification and validation utilities.
pub mod verification;

/// Web interface and HTTP API components.
pub mod web;

/// Web interface handlers and routes.
pub mod web_interface;

/// Core trait definitions for cross-crate integration.
///
/// Provides trait contracts used by optional companion crates:
/// - reasonkit-mem
/// - reasonkit-web
pub mod traits;

/// Aesthetic Expression Mastery System - M2-Enhanced UI/UX Assessment.
///
/// Leverages VIBE Benchmark Excellence (91.5% Web, 89.7% Android, 88.0% iOS)
/// for automated UI/UX quality assessment.
#[cfg(feature = "aesthetic")]
pub mod aesthetic;

/// VIBE Protocol Validation System.
///
/// Implements the revolutionary "Agent-as-a-Verifier" paradigm for
/// validating AI outputs against structured protocols.
#[cfg(feature = "vibe")]
pub mod vibe;

/// Multi-Language Code Intelligence Enhancement.
///
/// Provides code parsing, analysis, and understanding capabilities
/// across multiple programming languages.
#[cfg(feature = "code-intelligence")]
pub mod code_intelligence;

// ============================================================================
// MEMORY MODULES (optional - enable with `memory` feature)
// ============================================================================

/// Memory interface trait for reasonkit-mem integration.
///
/// Defines how reasonkit-core communicates with the reasonkit-mem
/// crate for storage, retrieval, and embedding operations.
pub mod memory_interface;

/// Re-export reasonkit-mem types when memory feature is enabled.
#[cfg(feature = "memory")]
pub use reasonkit_mem;

/// Re-export commonly used types from reasonkit-mem for convenience.
#[cfg(feature = "memory")]
pub use reasonkit_mem::{
    embedding, indexing, raptor, retrieval, storage, Error as MemError, Result as MemResult,
};

/// RAG (Retrieval-Augmented Generation) engine with LLM integration.
///
/// Provides the full RAG pipeline including document retrieval,
/// context augmentation, and LLM-powered generation.
#[cfg(feature = "memory")]
pub mod rag;

/// Autonomous Reasoning Framework for self-directed AI operations.
#[cfg(feature = "arf")]
pub mod arf;

/// GLM-4.6 model integration for agentic coordination and cost-efficient reasoning.
#[cfg(feature = "glm46")]
pub mod glm46;

// ============================================================================
// RE-EXPORTS
// ============================================================================

pub use error::{Error, Result};

/// Crate version string for runtime logging and API responses.
///
/// # Example
///
/// ```rust
/// println!("ReasonKit Core v{}", reasonkit::VERSION);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export orchestration system types
pub use orchestration::{
    ComponentCoordinator, ErrorRecovery, LongHorizonConfig, LongHorizonOrchestrator,
    LongHorizonResult, PerformanceTracker, StateManager, TaskGraph, TaskNode, TaskPriority,
    TaskStatus,
};

// Re-export engine module types
pub use engine::{
    Decision, MemoryContext, Profile as ReasoningProfile, ReasoningConfig, ReasoningError,
    ReasoningEvent, ReasoningLoop, ReasoningLoopBuilder, ReasoningSession, ReasoningStep, StepKind,
    StreamHandle, ThinkToolResult,
};

// Re-export Python bindings types for convenience
#[cfg(feature = "python")]
pub use bindings::{
    Profile as PyProfile, Reasoner as PyReasoner, ThinkToolOutput as PyThinkToolOutput,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Python module entry point (only when python feature is enabled)
#[cfg(feature = "python")]
mod python_module {
    #[allow(unused_imports)] // Used when python feature is enabled
    use super::*;
    use pyo3::prelude::*;

    /// Python module entry point for ReasonKit.
    ///
    /// This is the main entry point for the Python bindings, automatically
    /// called when the module is imported in Python.
    ///
    /// # Building
    ///
    /// ```bash
    /// cd reasonkit-core
    /// maturin develop --release   # Development install
    /// maturin build --release     # Build wheel for distribution
    /// ```
    ///
    /// # Python Usage
    ///
    /// ```python
    /// from reasonkit import Reasoner, Profile, ReasonerError
    /// from reasonkit import run_gigathink, run_laserlogic, run_bedrock
    /// from reasonkit import run_proofguard, run_brutalhonesty
    /// from reasonkit import quick_think, balanced_think, deep_think, paranoid_think
    /// from reasonkit import version
    ///
    /// # Check version
    /// print(f"ReasonKit v{version()}")
    ///
    /// # Create reasoner (auto-detects LLM from environment)
    /// r = Reasoner(use_mock=False)
    ///
    /// # Run individual ThinkTools
    /// result = r.run_gigathink("What factors drive startup success?")
    /// for perspective in result.perspectives():
    ///     print(f"- {perspective}")
    ///
    /// # Run with profile for comprehensive analysis
    /// result = r.think_with_profile(Profile.Balanced, "Should we use microservices?")
    /// print(f"Confidence: {result.confidence:.1%}")
    ///
    /// # Convenience functions (no Reasoner instantiation needed)
    /// result = run_gigathink("Analyze market trends", use_mock=True)
    /// result = balanced_think("Complex decision to make")
    /// ```
    #[pymodule]
    #[pyo3(name = "reasonkit")]
    fn reasonkit(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Register all bindings (classes, functions, exceptions)
        crate::bindings::register_bindings(m)?;
        Ok(())
    }
}

// ============================================================================
// CORE TYPES (always available - needed by ingestion, processing, etc.)
// ============================================================================

/// Document type categorization for the knowledge base.
///
/// Determines how documents are processed, indexed, and retrieved.
///
/// # Example
///
/// ```rust
/// use reasonkit::DocumentType;
///
/// let doc_type = DocumentType::Paper;
/// assert!(matches!(doc_type, DocumentType::Paper));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    /// Academic paper or research article (arXiv, journals, etc.)
    Paper,
    /// Technical documentation (API docs, guides, manuals)
    Documentation,
    /// Source code or code snippets
    Code,
    /// Personal notes or annotations
    Note,
    /// Transcript of audio/video content
    Transcript,
    /// Benchmark results or performance data
    Benchmark,
}

/// Source type enumeration for document provenance.
///
/// Tracks where documents originated for citation and verification.
///
/// # Example
///
/// ```rust
/// use reasonkit::SourceType;
///
/// let source = SourceType::Github;
/// assert!(matches!(source, SourceType::Github));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// arXiv preprint server
    Arxiv,
    /// GitHub repository
    Github,
    /// General website
    Website,
    /// Local file system
    Local,
    /// External API
    Api,
}

/// Source information for a document.
///
/// Contains provenance data including URLs, timestamps, and version information
/// for proper citation and retrieval tracking.
///
/// # Example
///
/// ```rust
/// use reasonkit::{Source, SourceType};
/// use chrono::Utc;
///
/// let source = Source {
///     source_type: SourceType::Github,
///     url: Some("https://github.com/org/repo".to_string()),
///     path: None,
///     arxiv_id: None,
///     github_repo: Some("org/repo".to_string()),
///     retrieved_at: Utc::now(),
///     version: Some("v1.0.0".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Type of source (determines how to interpret other fields)
    #[serde(rename = "type")]
    pub source_type: SourceType,

    /// URL of the source document (if applicable)
    pub url: Option<String>,

    /// Local file path (for local sources)
    pub path: Option<String>,

    /// arXiv paper ID (e.g., "2301.12345")
    pub arxiv_id: Option<String>,

    /// GitHub repository identifier (e.g., "owner/repo")
    pub github_repo: Option<String>,

    /// Timestamp when the document was retrieved
    pub retrieved_at: DateTime<Utc>,

    /// Version or commit hash of the source
    pub version: Option<String>,
}

/// Author information for document metadata.
///
/// # Example
///
/// ```rust
/// use reasonkit::Author;
///
/// let author = Author {
///     name: "Jane Doe".to_string(),
///     affiliation: Some("University of AI".to_string()),
///     email: Some("jane@example.com".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// Full name of the author
    pub name: String,

    /// Institutional affiliation
    pub affiliation: Option<String>,

    /// Contact email
    pub email: Option<String>,
}

/// Document metadata for indexing and retrieval.
///
/// Contains bibliographic information, tags, and categorization data
/// for rich document search and filtering.
///
/// # Example
///
/// ```rust
/// use reasonkit::Metadata;
///
/// let metadata = Metadata {
///     title: Some("Understanding AI Reasoning".to_string()),
///     authors: vec![],
///     abstract_text: Some("This paper explores...".to_string()),
///     tags: vec!["ai".to_string(), "reasoning".to_string()],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Document title
    pub title: Option<String>,

    /// List of authors
    pub authors: Vec<Author>,

    /// Abstract or summary text
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,

    /// Publication date (ISO 8601 format)
    pub date: Option<String>,

    /// Publication venue (journal, conference, etc.)
    pub venue: Option<String>,

    /// Citation count (if available)
    pub citations: Option<i32>,

    /// User-defined tags
    pub tags: Vec<String>,

    /// Subject categories
    pub categories: Vec<String>,

    /// Extracted keywords
    pub keywords: Vec<String>,

    /// Digital Object Identifier
    pub doi: Option<String>,

    /// License information
    pub license: Option<String>,
}

/// References to different embedding types for a chunk.
///
/// Supports hybrid retrieval by tracking multiple embedding representations
/// (dense, sparse, ColBERT) for each text chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingIds {
    /// Dense embedding ID (e.g., from OpenAI, Cohere)
    pub dense: Option<String>,

    /// Sparse embedding ID (e.g., BM25, SPLADE)
    pub sparse: Option<String>,

    /// ColBERT multi-vector embedding ID
    pub colbert: Option<String>,
}

/// A chunk of text from a document.
///
/// Documents are split into chunks for embedding and retrieval.
/// Each chunk maintains positional information and embedding references.
///
/// # Example
///
/// ```rust
/// use reasonkit::{Chunk, EmbeddingIds};
/// use uuid::Uuid;
///
/// let chunk = Chunk {
///     id: Uuid::new_v4(),
///     text: "This is a chunk of text...".to_string(),
///     index: 0,
///     start_char: 0,
///     end_char: 26,
///     token_count: Some(7),
///     section: Some("Introduction".to_string()),
///     page: Some(1),
///     embedding_ids: EmbeddingIds::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk
    pub id: Uuid,

    /// The text content of the chunk
    pub text: String,

    /// Position index within the document
    pub index: usize,

    /// Starting character position in the original document
    pub start_char: usize,

    /// Ending character position in the original document
    pub end_char: usize,

    /// Estimated token count for the chunk
    pub token_count: Option<usize>,

    /// Section or heading this chunk belongs to
    pub section: Option<String>,

    /// Page number (for paginated documents)
    pub page: Option<usize>,

    /// References to stored embeddings
    pub embedding_ids: EmbeddingIds,
}

/// Processing state enumeration for documents.
///
/// Tracks the current state of a document in the processing pipeline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingState {
    /// Document is queued for processing
    #[default]
    Pending,
    /// Document is currently being processed
    Processing,
    /// Processing completed successfully
    Completed,
    /// Processing failed with errors
    Failed,
}

/// Processing status for a document.
///
/// Tracks which processing stages have been completed and any errors encountered.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatus {
    /// Current processing state
    pub status: ProcessingState,

    /// Whether the document has been chunked
    pub chunked: bool,

    /// Whether embeddings have been generated
    pub embedded: bool,

    /// Whether the document has been indexed
    pub indexed: bool,

    /// Whether RAPTOR summarization has been applied
    pub raptor_processed: bool,

    /// List of error messages (if any)
    pub errors: Vec<String>,
}

/// Content format enumeration.
///
/// Identifies the format of document content for proper parsing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentFormat {
    /// Plain text
    #[default]
    Text,
    /// Markdown format
    Markdown,
    /// HTML content
    Html,
    /// LaTeX source
    Latex,
}

/// Document content container.
///
/// Stores the raw content along with format and statistical information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentContent {
    /// Raw content string
    pub raw: String,

    /// Content format
    pub format: ContentFormat,

    /// Primary language code (e.g., "en", "zh")
    pub language: String,

    /// Word count
    pub word_count: usize,

    /// Character count
    pub char_count: usize,
}

/// A document in the knowledge base.
///
/// The primary data structure for storing and managing documents.
/// Contains content, metadata, processing status, and chunks.
///
/// # Example
///
/// ```rust
/// use reasonkit::{Document, DocumentType, Source, SourceType};
/// use chrono::Utc;
///
/// let source = Source {
///     source_type: SourceType::Local,
///     url: None,
///     path: Some("/path/to/doc.md".to_string()),
///     arxiv_id: None,
///     github_repo: None,
///     retrieved_at: Utc::now(),
///     version: None,
/// };
///
/// let doc = Document::new(DocumentType::Documentation, source)
///     .with_content("# My Document\n\nContent here...".to_string());
///
/// assert_eq!(doc.doc_type, DocumentType::Documentation);
/// assert!(doc.content.word_count > 0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: Uuid,

    /// Document type categorization
    #[serde(rename = "type")]
    pub doc_type: DocumentType,

    /// Source information for provenance
    pub source: Source,

    /// Document content
    pub content: DocumentContent,

    /// Document metadata
    pub metadata: Metadata,

    /// Processing status
    pub processing: ProcessingStatus,

    /// Text chunks for retrieval
    pub chunks: Vec<Chunk>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: Option<DateTime<Utc>>,
}

impl Document {
    /// Create a new document with the given type and source.
    ///
    /// # Arguments
    ///
    /// * `doc_type` - The type of document
    /// * `source` - Source information for provenance
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit::{Document, DocumentType, Source, SourceType};
    /// use chrono::Utc;
    ///
    /// let source = Source {
    ///     source_type: SourceType::Local,
    ///     url: None,
    ///     path: Some("/path/to/file.txt".to_string()),
    ///     arxiv_id: None,
    ///     github_repo: None,
    ///     retrieved_at: Utc::now(),
    ///     version: None,
    /// };
    ///
    /// let doc = Document::new(DocumentType::Note, source);
    /// assert_eq!(doc.doc_type, DocumentType::Note);
    /// ```
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

    /// Set the document content and compute statistics.
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw content string
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit::{Document, DocumentType, Source, SourceType};
    /// use chrono::Utc;
    ///
    /// let source = Source {
    ///     source_type: SourceType::Local,
    ///     url: None,
    ///     path: None,
    ///     arxiv_id: None,
    ///     github_repo: None,
    ///     retrieved_at: Utc::now(),
    ///     version: None,
    /// };
    ///
    /// let doc = Document::new(DocumentType::Note, source)
    ///     .with_content("Hello world".to_string());
    ///
    /// assert_eq!(doc.content.word_count, 2);
    /// assert_eq!(doc.content.char_count, 11);
    /// ```
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

    /// Set the document metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The metadata to set
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit::{Document, DocumentType, Source, SourceType, Metadata};
    /// use chrono::Utc;
    ///
    /// let source = Source {
    ///     source_type: SourceType::Local,
    ///     url: None,
    ///     path: None,
    ///     arxiv_id: None,
    ///     github_repo: None,
    ///     retrieved_at: Utc::now(),
    ///     version: None,
    /// };
    ///
    /// let metadata = Metadata {
    ///     title: Some("My Document".to_string()),
    ///     ..Default::default()
    /// };
    ///
    /// let doc = Document::new(DocumentType::Note, source)
    ///     .with_metadata(metadata);
    ///
    /// assert_eq!(doc.metadata.title, Some("My Document".to_string()));
    /// ```
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

/// Source of a search match for hybrid retrieval.
///
/// Indicates which retrieval method produced a search result,
/// enabling score fusion and result explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    /// Dense vector retrieval (semantic similarity)
    Dense,
    /// Sparse retrieval (BM25, keyword matching)
    Sparse,
    /// Hybrid retrieval (combined dense + sparse)
    Hybrid,
    /// RAPTOR hierarchical retrieval
    Raptor,
}

/// Search result from a query.
///
/// Contains the matched chunk, relevance score, and source information.
///
/// # Example
///
/// ```rust
/// use reasonkit::{SearchResult, MatchSource, Chunk, EmbeddingIds};
/// use uuid::Uuid;
///
/// let chunk = Chunk {
///     id: Uuid::new_v4(),
///     text: "Relevant content...".to_string(),
///     index: 0,
///     start_char: 0,
///     end_char: 19,
///     token_count: Some(2),
///     section: None,
///     page: None,
///     embedding_ids: EmbeddingIds::default(),
/// };
///
/// let result = SearchResult {
///     score: 0.95,
///     document_id: Uuid::new_v4(),
///     chunk,
///     match_source: MatchSource::Dense,
/// };
///
/// assert!(result.score > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Relevance score (higher is more relevant)
    pub score: f32,

    /// ID of the document containing the match
    pub document_id: Uuid,

    /// The matched chunk
    pub chunk: Chunk,

    /// Which retrieval method produced this match
    pub match_source: MatchSource,
}

// ============================================================================
// MEMORY-SPECIFIC TYPES (only with `memory` feature)
// ============================================================================

#[cfg(feature = "memory")]
pub use reasonkit_mem::RetrievalConfig;

/// Simple retrieval configuration (available without memory feature).
///
/// Provides basic retrieval parameters when the full memory layer is not enabled.
#[cfg(not(feature = "memory"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Maximum number of results to return
    pub top_k: usize,

    /// Minimum relevance score threshold
    pub min_score: f32,

    /// Weight for dense retrieval in hybrid mode (0.0-1.0)
    pub alpha: f32,

    /// Whether to use RAPTOR hierarchical retrieval
    pub use_raptor: bool,

    /// Whether to rerank results
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

    #[test]
    fn test_document_with_content() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: None,
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };
        let doc =
            Document::new(DocumentType::Note, source).with_content("Hello world test".to_string());

        assert_eq!(doc.content.word_count, 3);
        assert_eq!(doc.content.char_count, 16);
    }

    #[test]
    fn test_version_available() {
        assert!(!VERSION.is_empty());
    }
}
