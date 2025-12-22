//! Common types shared across all ReasonKit modules
//!
//! Provides error types, result aliases, and shared utilities.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════════
// ERROR TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Unified error type for ReasonKit operations
#[derive(Debug, Error)]
pub enum ReasonKitError {
    // ─────────────────────────────────────────────────────────────────────────
    // Configuration Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Configuration file not found: {path}")]
    ConfigNotFound { path: PathBuf },

    #[error("Invalid configuration: {field} - {reason}")]
    InvalidConfig { field: String, reason: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Ingestion Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("Unsupported format: {format}")]
    UnsupportedFormat { format: String },

    #[error("Parse error in {file}: {message}")]
    ParseError { file: String, message: String },

    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // ─────────────────────────────────────────────────────────────────────────
    // Embedding Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("Embedding provider error: {provider} - {message}")]
    EmbeddingProvider { provider: String, message: String },

    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Rate limit exceeded for {provider}")]
    RateLimitExceeded { provider: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Retrieval Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("Index not found: {name}")]
    IndexNotFound { name: String },

    #[error("Search error: {message}")]
    SearchError { message: String },

    #[error("Vector store error: {message}")]
    VectorStoreError { message: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Orchestration Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("No capable agent available for task: {task_id}")]
    NoCapableAgent { task_id: String },

    #[error("Task timeout exceeded: {task_id}")]
    TaskTimeout { task_id: String },

    #[error("Agent unavailable: {agent_id}")]
    AgentUnavailable { agent_id: String },

    #[error("Escalation failed: {reason}")]
    EscalationFailed { reason: String },

    // ─────────────────────────────────────────────────────────────────────────
    // API Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Authentication failed: {message}")]
    AuthError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    // ─────────────────────────────────────────────────────────────────────────
    // Generic Errors
    // ─────────────────────────────────────────────────────────────────────────
    #[error("Internal error: {message}")]
    Internal { message: String },

    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    #[error("{0}")]
    Custom(String),
}

/// Result type alias for ReasonKit operations
pub type Result<T> = std::result::Result<T, ReasonKitError>;

// ═══════════════════════════════════════════════════════════════════════════
// COMMON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub String);

impl Id {
    /// Create a new ID with prefix
    pub fn new(prefix: &str) -> Self {
        use chrono::Utc;
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let random: u32 = rand::random::<u32>() % 0xFFFF;
        Self(format!("{}_{:?}_{:04x}", prefix, timestamp, random))
    }

    /// Create from existing string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Id {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Confidence score (0.0 - 1.0)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new confidence score, clamping to [0.0, 1.0]
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the inner value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if confidence meets threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.0 >= threshold
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(0.5)
    }
}

/// Token count and cost tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub cost_usd: f64,
}

impl TokenUsage {
    pub fn new(input: u32, output: u32, cost: f64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cost_usd: cost,
        }
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cost_usd += other.cost_usd;
    }
}

/// Metadata key-value pairs
pub type Metadata = std::collections::HashMap<String, serde_json::Value>;

/// Source attribution for provenance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub path: Option<PathBuf>,
    pub url: Option<String>,
    pub version: Option<String>,
    pub retrieved_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generation() {
        let id1 = Id::new("task");
        let id2 = Id::new("task");
        assert!(id1.0.starts_with("task_"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_confidence_clamping() {
        assert_eq!(Confidence::new(1.5).value(), 1.0);
        assert_eq!(Confidence::new(-0.5).value(), 0.0);
        assert_eq!(Confidence::new(0.75).value(), 0.75);
    }

    #[test]
    fn test_token_usage_add() {
        let mut usage1 = TokenUsage::new(100, 200, 0.01);
        let usage2 = TokenUsage::new(50, 100, 0.005);
        usage1.add(&usage2);
        assert_eq!(usage1.input_tokens, 150);
        assert_eq!(usage1.output_tokens, 300);
        assert_eq!(usage1.total_tokens, 450);
    }
}
