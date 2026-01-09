//! Error types for ReasonKit Core.
//!
//! This module defines the error types used throughout ReasonKit Core.
//! All public functions return [`Result<T>`] which is an alias for
//! `std::result::Result<T, Error>`.
//!
//! # Error Handling Philosophy
//!
//! ReasonKit follows these error handling principles:
//!
//! 1. **Fail Fast, Fail Loud** - Errors are surfaced immediately with actionable messages
//! 2. **Typed Errors** - Each error variant carries specific context for the failure mode
//! 3. **Error Chaining** - Use `with_context()` to add context while preserving the source
//!
//! # Example
//!
//! ```rust
//! use reasonkit_core::{Error, Result};
//!
//! fn process_document(path: &str) -> Result<()> {
//!     // Return typed errors
//!     if path.is_empty() {
//!         return Err(Error::Validation("Path cannot be empty".to_string()));
//!     }
//!
//!     // Create errors with context
//!     std::fs::read_to_string(path)
//!         .map_err(|e| Error::io(format!("Failed to read {}", path)))?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Error Categories
//!
//! | Category | Variants | Typical Cause |
//! |----------|----------|---------------|
//! | I/O | `Io`, `IoMessage` | File operations, network I/O |
//! | Parsing | `Json`, `Parse` | Malformed input, schema violations |
//! | Resource | `NotFound`, `DocumentNotFound` | Missing documents, chunks |
//! | Processing | `Embedding`, `Indexing`, `Retrieval` | Pipeline failures |
//! | Configuration | `Config`, `ConfigError` | Invalid settings |
//! | M2 Integration | `M2*` variants | MiniMax M2 API issues |

use thiserror::Error;

/// Result type alias for ReasonKit operations.
///
/// All fallible operations in ReasonKit return this type.
///
/// # Example
///
/// ```rust
/// use reasonkit_core::Result;
///
/// fn do_something() -> Result<String> {
///     Ok("Success".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for ReasonKit Core.
///
/// This enum encompasses all error conditions that can occur during
/// ReasonKit operations. Each variant carries context-specific information
/// to aid in debugging and error recovery.
///
/// # Example
///
/// ```rust
/// use reasonkit_core::Error;
///
/// // Create specific error types
/// let validation_err = Error::Validation("Invalid input".to_string());
/// let not_found_err = Error::NotFound { resource: "document:123".to_string() };
///
/// // Add context to existing errors
/// let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
/// let wrapped = Error::with_context("Loading config", io_error);
/// ```
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error from standard library operations.
    ///
    /// Automatically converted from `std::io::Error`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    ///
    /// Automatically converted from `serde_json::Error`.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// PDF processing error.
    ///
    /// Occurs during PDF parsing, text extraction, or page processing.
    #[error("PDF processing error: {0}")]
    Pdf(String),

    /// Document not found in the knowledge base.
    ///
    /// The document ID does not exist or has been deleted.
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Chunk not found within a document.
    ///
    /// The chunk ID does not exist or the document has been re-chunked.
    #[error("Chunk not found: {0}")]
    ChunkNotFound(String),

    /// Embedding generation or retrieval error.
    ///
    /// Occurs during vector embedding operations.
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Indexing operation error.
    ///
    /// Occurs during document indexing or index updates.
    #[error("Indexing error: {0}")]
    Indexing(String),

    /// Retrieval operation error.
    ///
    /// Occurs during document or chunk retrieval.
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// Storage operation error.
    ///
    /// Occurs during database or file storage operations.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Configuration error.
    ///
    /// Invalid or missing configuration values.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network/HTTP error.
    ///
    /// Occurs during API calls, downloads, or network operations.
    #[error("Network error: {0}")]
    Network(String),

    /// MCP Daemon error.
    ///
    /// Occurs during daemon lifecycle management (start, stop, IPC).
    #[error("Daemon error: {0}")]
    Daemon(String),

    /// Schema validation error.
    ///
    /// Input data does not conform to expected schema.
    #[error("Schema validation error: {0}")]
    Validation(String),

    /// Generic resource not found error.
    ///
    /// Use this when a resource type is dynamic or not covered by specific variants.
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// let err = Error::NotFound { resource: "protocol:gigathink".to_string() };
    /// assert!(err.to_string().contains("gigathink"));
    /// ```
    #[error("{resource} not found")]
    NotFound {
        /// Resource identifier that was not found
        resource: String,
    },

    /// I/O error with custom message.
    ///
    /// Use when you need to provide additional context beyond what
    /// the standard I/O error contains.
    #[error("I/O error: {message}")]
    IoMessage {
        /// Descriptive error message
        message: String,
    },

    /// Parse error with custom message.
    ///
    /// Use for parsing failures that aren't JSON-specific.
    #[error("Parse error: {message}")]
    Parse {
        /// Descriptive error message
        message: String,
    },

    /// Qdrant vector database error.
    ///
    /// Occurs during Qdrant operations.
    #[error("Qdrant error: {0}")]
    Qdrant(String),

    /// Tantivy search engine error.
    ///
    /// Occurs during Tantivy indexing or search operations.
    #[error("Tantivy error: {0}")]
    Tantivy(String),

    /// ARF (Autonomous Reasoning Framework) error.
    ///
    /// Occurs during autonomous reasoning operations.
    #[cfg(feature = "arf")]
    #[error("ARF error: {0}")]
    ArfError(String),

    /// Config library parsing error.
    ///
    /// Occurs when configuration files cannot be parsed.
    #[error("Config library error: {0}")]
    ConfigError(String),

    /// Memory module error from reasonkit-mem.
    ///
    /// Converted from `reasonkit_mem::Error`.
    #[cfg(feature = "memory")]
    #[error("Memory error: {0}")]
    Memory(String),

    /// Generic error with context chain.
    ///
    /// Use [`Error::with_context()`] to create this variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    /// let err = Error::with_context("Loading configuration file", io_err);
    /// assert!(err.to_string().contains("Loading configuration"));
    /// ```
    #[error("{context}: {source}")]
    WithContext {
        /// Description of what operation was being attempted
        context: String,
        /// The underlying error
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// MCP (Model Context Protocol) error.
    ///
    /// Occurs during MCP server/client operations.
    #[error("MCP error: {0}")]
    Mcp(String),

    /// M2 API execution error.
    ///
    /// Occurs during MiniMax M2 model API calls.
    #[error("M2 execution error: {0}")]
    M2ExecutionError(String),

    /// M2 rate limit exceeded.
    ///
    /// The API rate limit has been exceeded. Retry after waiting.
    #[error("M2 rate limit exceeded")]
    RateLimitExceeded,

    /// M2 budget exceeded.
    ///
    /// The configured budget limit has been reached.
    /// Contains (actual_cost, budget_limit).
    #[error("M2 budget exceeded: {0} > {1}")]
    BudgetExceeded(f64, f64),

    /// M2 protocol validation error.
    ///
    /// The protocol definition or output failed validation.
    #[error("M2 protocol validation error: {0}")]
    M2ProtocolValidation(String),

    /// M2 constraint violation.
    ///
    /// A protocol constraint was not satisfied.
    #[error("M2 constraint violation: {0}")]
    M2ConstraintViolation(String),

    /// M2 framework incompatibility.
    ///
    /// The requested operation is not compatible with the current framework.
    #[error("M2 framework incompatibility: {0}")]
    M2FrameworkIncompatibility(String),

    /// Resource exhausted (memory, connections, etc.).
    ///
    /// A system resource has been exhausted.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Template not found.
    ///
    /// The requested protocol or profile template does not exist.
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    /// Code intelligence error.
    ///
    /// Occurs during code parsing or analysis.
    #[error("Code intelligence error: {0}")]
    CodeIntelligence(String),

    /// M2 integration error.
    ///
    /// General M2 integration failure.
    #[error("M2 integration error: {0}")]
    M2IntegrationError(String),

    /// Timeout error.
    ///
    /// An operation exceeded its time limit.
    #[error("Timeout error: {0}")]
    Timeout(String),

    /// Dependency not met.
    ///
    /// A required dependency (step, component, etc.) was not satisfied.
    #[error("Dependency not met: {0}")]
    DependencyNotMet(String),

    /// Protocol generation error.
    ///
    /// Failed to generate a protocol definition.
    #[error("Protocol generation error: {0}")]
    ProtocolGenerationError(String),

    /// ThinkTool execution error.
    ///
    /// A ThinkTool failed during execution.
    #[error("ThinkTool execution error: {0}")]
    ThinkToolExecutionError(String),

    /// Rate limit error.
    ///
    /// API rate limiting was triggered.
    #[error("Rate limit error: {0}")]
    RateLimit(String),

    /// Authentication error.
    ///
    /// Authentication failed (invalid or missing credentials).
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Authorization error.
    ///
    /// The authenticated user lacks permission for the operation.
    #[error("Authorization error: {0}")]
    Authorization(String),
}

impl Error {
    /// Create an I/O error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// let err = Error::io("Failed to open file");
    /// assert!(err.to_string().contains("Failed to open"));
    /// ```
    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(std::io::Error::other(msg.into()))
    }

    /// Create a parse error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// let err = Error::parse("Invalid JSON syntax at line 42");
    /// ```
    pub fn parse(msg: impl Into<String>) -> Self {
        // Since serde_json::Error doesn't have a simple constructor,
        // we use io error to create it
        let io_err = std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into());
        Self::Json(serde_json::Error::io(io_err))
    }

    /// Create a PDF processing error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn pdf(msg: impl Into<String>) -> Self {
        Self::Pdf(msg.into())
    }

    /// Create an embedding error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }

    /// Create an indexing error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn indexing(msg: impl Into<String>) -> Self {
        Self::Indexing(msg.into())
    }

    /// Create a retrieval error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn retrieval(msg: impl Into<String>) -> Self {
        Self::Retrieval(msg.into())
    }

    /// Create a query error from a message.
    ///
    /// This is an alias for [`retrieval()`](Self::retrieval).
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn query(msg: impl Into<String>) -> Self {
        Self::Retrieval(msg.into())
    }

    /// Create a storage error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Create a config error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a network error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network(msg.into())
    }

    /// Create a daemon error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    pub fn daemon(msg: impl Into<String>) -> Self {
        Self::Daemon(msg.into())
    }

    /// Create a validation error from a message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// let err = Error::validation("Input must not be empty");
    /// ```
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create an ARF error from a message.
    ///
    /// Only available with the `arf` feature.
    ///
    /// # Arguments
    ///
    /// * `msg` - Descriptive error message
    #[cfg(feature = "arf")]
    pub fn arf_error(msg: impl Into<String>) -> Self {
        Self::ArfError(msg.into())
    }

    /// Wrap an error with additional context.
    ///
    /// This creates an error chain, preserving the original error
    /// while adding context about what operation was being attempted.
    ///
    /// # Arguments
    ///
    /// * `context` - Description of the operation that failed
    /// * `source` - The underlying error
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_core::Error;
    ///
    /// fn load_config(path: &str) -> Result<(), Error> {
    ///     std::fs::read_to_string(path)
    ///         .map_err(|e| Error::with_context(
    ///             format!("Loading config from {}", path),
    ///             e
    ///         ))?;
    ///     Ok(())
    /// }
    /// ```
    pub fn with_context<E>(context: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::WithContext {
            context: context.into(),
            source: Box::new(source),
        }
    }
}

/// Convert reqwest errors to our Error type.
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Self::Network(err.to_string())
    }
}

/// Convert config::ConfigError to our Error type.
#[cfg(feature = "arf")]
impl From<config::ConfigError> for Error {
    fn from(err: config::ConfigError) -> Self {
        Self::ConfigError(err.to_string())
    }
}

/// Convert sled errors to our Error type.
#[cfg(feature = "arf")]
impl From<sled::Error> for Error {
    fn from(err: sled::Error) -> Self {
        Self::Storage(err.to_string())
    }
}

/// Convert anyhow errors to our Error type.
#[cfg(feature = "arf")]
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::arf_error(err.to_string())
    }
}

/// Convert mpsc send errors to our Error type.
#[cfg(feature = "arf")]
impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(err: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::arf_error(err.to_string())
    }
}

/// Convert reasonkit-mem errors to core Error type.
#[cfg(feature = "memory")]
impl From<reasonkit_mem::MemError> for Error {
    fn from(err: reasonkit_mem::MemError) -> Self {
        Error::Memory(err.to_string())
    }
}

/// Extension trait for adding context to Results.
///
/// This trait provides a convenient way to add context to any `Result`
/// type, similar to `anyhow::Context`.
///
/// # Example
///
/// ```rust
/// use reasonkit_core::{Result, ResultExt};
///
/// fn load_data(path: &str) -> Result<String> {
///     std::fs::read_to_string(path)
///         .context(format!("Failed to load data from {}", path))
/// }
/// ```
pub trait ResultExt<T> {
    /// Add context to an error.
    ///
    /// If the result is `Ok`, it is returned unchanged.
    /// If the result is `Err`, the error is wrapped with the given context.
    ///
    /// # Arguments
    ///
    /// * `context` - Description of the operation that was being attempted
    fn context(self, context: impl Into<String>) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::with_context(context, e))
    }
}
