//! Error types for ReasonKit Core

use thiserror::Error;

/// Result type alias for ReasonKit operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for ReasonKit Core
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// PDF processing error
    #[error("PDF processing error: {0}")]
    Pdf(String),

    /// Document not found
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Chunk not found
    #[error("Chunk not found: {0}")]
    ChunkNotFound(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Indexing error
    #[error("Indexing error: {0}")]
    Indexing(String),

    /// Retrieval error
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network/HTTP error
    #[error("Network error: {0}")]
    Network(String),

    /// Schema validation error
    #[error("Schema validation error: {0}")]
    Validation(String),

    /// Resource not found
    #[error("{resource} not found")]
    NotFound {
        /// Resource that was not found
        resource: String,
    },

    /// I/O error with message
    #[error("I/O error: {message}")]
    IoMessage {
        /// Error message
        message: String,
    },

    /// Parse error with message
    #[error("Parse error: {message}")]
    Parse {
        /// Error message
        message: String,
    },

    /// Qdrant client error
    #[error("Qdrant error: {0}")]
    Qdrant(String),

    /// Tantivy search error
    #[error("Tantivy error: {0}")]
    Tantivy(String),

    /// ARF (Autonomous Reasoning Framework) error
    #[cfg(feature = "arf")]
    #[error("ARF error: {0}")]
    ArfError(String),

    /// Config library error
    #[error("Config library error: {0}")]
    ConfigError(String),

    /// Memory module error (from reasonkit-mem)
    #[cfg(feature = "memory")]
    #[error("Memory error: {0}")]
    Memory(String),

    /// Generic error with context
    #[error("{context}: {source}")]
    WithContext {
        /// Error context
        context: String,
        /// Original error
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// MCP error
    #[error("MCP error: {0}")]
    Mcp(String),

    /// M2 API execution error
    #[error("M2 execution error: {0}")]
    M2ExecutionError(String),

    /// M2 rate limit exceeded
    #[error("M2 rate limit exceeded")]
    RateLimitExceeded,

    /// M2 budget exceeded
    #[error("M2 budget exceeded: {0} > {1}")]
    BudgetExceeded(f64, f64),

    /// M2 protocol validation error
    #[error("M2 protocol validation error: {0}")]
    M2ProtocolValidation(String),

    /// M2 constraint violation
    #[error("M2 constraint violation: {0}")]
    M2ConstraintViolation(String),

    /// M2 framework incompatibility
    #[error("M2 framework incompatibility: {0}")]
    M2FrameworkIncompatibility(String),

    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Template not found
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    /// Code intelligence error
    #[error("Code intelligence error: {0}")]
    CodeIntelligence(String),

    /// M2 integration error
    #[error("M2 integration error: {0}")]
    M2IntegrationError(String),

    /// Timeout error
    #[error("Timeout error: {0}")]
    Timeout(String),

    /// Dependency not met error
    #[error("Dependency not met: {0}")]
    DependencyNotMet(String),

    /// Protocol generation error
    #[error("Protocol generation error: {0}")]
    ProtocolGenerationError(String),

    /// ThinkTool execution error
    #[error("ThinkTool execution error: {0}")]
    ThinkToolExecutionError(String),

    /// Rate limit error
    #[error("Rate limit error: {0}")]
    RateLimit(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Authorization error
    #[error("Authorization error: {0}")]
    Authorization(String),
}

impl Error {
    /// Create an I/O error
    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(std::io::Error::other(msg.into()))
    }

    /// Create a parse error (JSON)
    pub fn parse(msg: impl Into<String>) -> Self {
        // Since serde_json::Error doesn't have a simple constructor,
        // we use io error to create it
        let io_err = std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into());
        Self::Json(serde_json::Error::io(io_err))
    }

    /// Create a PDF processing error
    pub fn pdf(msg: impl Into<String>) -> Self {
        Self::Pdf(msg.into())
    }

    /// Create an embedding error
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }

    /// Create an indexing error
    pub fn indexing(msg: impl Into<String>) -> Self {
        Self::Indexing(msg.into())
    }

    /// Create a retrieval error
    pub fn retrieval(msg: impl Into<String>) -> Self {
        Self::Retrieval(msg.into())
    }

    /// Create a query error (alias for retrieval)
    pub fn query(msg: impl Into<String>) -> Self {
        Self::Retrieval(msg.into())
    }

    /// Create a storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create an ARF error
    #[cfg(feature = "arf")]
    pub fn arf_error(msg: impl Into<String>) -> Self {
        Self::ArfError(msg.into())
    }

    /// Wrap an error with context
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

/// Convert reqwest errors to our Error type
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Self::Network(err.to_string())
    }
}

/// Convert config::ConfigError to our Error type
#[cfg(feature = "arf")]
impl From<config::ConfigError> for Error {
    fn from(err: config::ConfigError) -> Self {
        Self::ConfigError(err.to_string())
    }
}

/// Convert sled errors to our Error type
#[cfg(feature = "arf")]
impl From<sled::Error> for Error {
    fn from(err: sled::Error) -> Self {
        Self::Storage(err.to_string())
    }
}

/// Convert anyhow errors to our Error type
#[cfg(feature = "arf")]
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::arf_error(err.to_string())
    }
}

/// Convert mpsc send errors to our Error type
#[cfg(feature = "arf")]
impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(err: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::arf_error(err.to_string())
    }
}

/// Convert reasonkit-mem errors to core Error type
#[cfg(feature = "memory")]
impl From<reasonkit_mem::MemError> for Error {
    fn from(err: reasonkit_mem::MemError) -> Self {
        Error::Memory(err.to_string())
    }
}

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    /// Add context to an error
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
