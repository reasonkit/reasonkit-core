//! Error types for ARF Core

use thiserror::Error;

/// ARF Error type
#[derive(Error, Debug)]
pub enum ArfError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Plugin error: {0}")]
    Plugin(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("State error: {0}")]
    State(String),
}

/// Result type alias for ARF operations
pub type Result<T> = std::result::Result<T, ArfError>;
