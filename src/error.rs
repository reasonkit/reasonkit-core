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

    /// Generic error with context
    #[error("{context}: {source}")]
    WithContext {
        /// Error context
        context: String,
        /// Original error
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
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

/// Convert config::ConfigError to our Error type
#[cfg(feature = "arf")]
impl From<config::ConfigError> for Error {
    fn from(err: config::ConfigError) -> Self {
        Self::ConfigError(err.to_string())
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
