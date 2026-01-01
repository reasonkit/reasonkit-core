//! # LLM CLI Ecosystem Integration
//!
//! Integrates Simon Willison's LLM CLI tool for multi-model orchestration,
//! embeddings, clustering, and RAG pipelines.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::llm_cli::{LlmCliClient, EmbeddingConfig, ClusterConfig};
//!
//! let client = LlmCliClient::new()?;
//!
//! // Execute a prompt
//! let response = client.prompt("Analyze this code", Some("claude-sonnet-4")).await?;
//!
//! // Generate embeddings
//! let embeddings = client.embed("text to embed", None).await?;
//!
//! // Cluster documents
//! let clusters = client.cluster("documents", 5, None).await?;
//! ```
//!
//! ## Security
//!
//! All user-provided inputs are validated and sanitized before being passed to
//! shell commands. This includes:
//! - Model names (alphanumeric, hyphens, underscores, slashes, colons, dots)
//! - Collection names (alphanumeric, hyphens, underscores)
//! - Database paths (validated as safe filesystem paths)
//! - Template names (alphanumeric, hyphens, underscores)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

use crate::error::{Error, Result};

/// Maximum length for user-provided string inputs to prevent DoS
const MAX_INPUT_LENGTH: usize = 10_000;

/// Maximum length for identifiers (model names, collection names, etc.)
const MAX_IDENTIFIER_LENGTH: usize = 256;

/// Validate and sanitize a model name.
/// Allowed characters: alphanumeric, hyphens, underscores, slashes, colons, dots.
/// Examples: "gpt-4o-mini", "claude-sonnet-4", "sentence-transformers/all-MiniLM-L6-v2"
fn validate_model_name(model: &str) -> Result<&str> {
    if model.is_empty() {
        return Err(Error::Validation("Model name cannot be empty".to_string()));
    }
    if model.len() > MAX_IDENTIFIER_LENGTH {
        return Err(Error::Validation(format!(
            "Model name exceeds maximum length of {} characters",
            MAX_IDENTIFIER_LENGTH
        )));
    }
    // Allow alphanumeric, hyphens, underscores, slashes, colons, dots
    // This covers: gpt-4o-mini, claude-3, sentence-transformers/all-MiniLM-L6-v2
    if !model
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '/' || c == ':' || c == '.')
    {
        return Err(Error::Validation(format!(
            "Model name contains invalid characters: '{}'. Allowed: alphanumeric, -, _, /, :, .",
            model
        )));
    }
    // Prevent path traversal attempts
    if model.contains("..") {
        return Err(Error::Validation(
            "Model name cannot contain '..' (path traversal)".to_string(),
        ));
    }
    Ok(model)
}

/// Validate a collection name.
/// Allowed characters: alphanumeric, hyphens, underscores.
fn validate_collection_name(collection: &str) -> Result<&str> {
    if collection.is_empty() {
        return Err(Error::Validation(
            "Collection name cannot be empty".to_string(),
        ));
    }
    if collection.len() > MAX_IDENTIFIER_LENGTH {
        return Err(Error::Validation(format!(
            "Collection name exceeds maximum length of {} characters",
            MAX_IDENTIFIER_LENGTH
        )));
    }
    // Collection names should be simple identifiers
    if !collection
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(Error::Validation(format!(
            "Collection name contains invalid characters: '{}'. Allowed: alphanumeric, -, _",
            collection
        )));
    }
    Ok(collection)
}

/// Validate a template name.
/// Allowed characters: alphanumeric, hyphens, underscores.
fn validate_template_name(template: &str) -> Result<&str> {
    if template.is_empty() {
        return Err(Error::Validation(
            "Template name cannot be empty".to_string(),
        ));
    }
    if template.len() > MAX_IDENTIFIER_LENGTH {
        return Err(Error::Validation(format!(
            "Template name exceeds maximum length of {} characters",
            MAX_IDENTIFIER_LENGTH
        )));
    }
    if !template
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(Error::Validation(format!(
            "Template name contains invalid characters: '{}'. Allowed: alphanumeric, -, _",
            template
        )));
    }
    Ok(template)
}

/// Format a dangerous character for display in error messages.
fn format_dangerous_char(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\0' => "\\0".to_string(),
        _ => c.to_string(),
    }
}

/// Validate a database path.
/// Must be a valid path without shell metacharacters.
fn validate_db_path(path: &str) -> Result<&str> {
    if path.is_empty() {
        return Err(Error::Validation(
            "Database path cannot be empty".to_string(),
        ));
    }
    if path.len() > 4096 {
        return Err(Error::Validation(
            "Database path exceeds maximum length".to_string(),
        ));
    }
    // Reject shell metacharacters and control characters
    // Safe characters: alphanumeric, path separators, dots, hyphens, underscores
    let dangerous_chars = [
        '$', '`', '!', '&', '|', ';', '(', ')', '{', '}', '<', '>', '\n', '\r', '\0', '"', '\'',
        '\\',
    ];
    for c in dangerous_chars {
        if path.contains(c) {
            return Err(Error::Validation(format!(
                "Database path contains dangerous character: '{}'",
                format_dangerous_char(c)
            )));
        }
    }
    // Prevent path traversal
    if path.contains("..") {
        return Err(Error::Validation(
            "Database path cannot contain '..' (path traversal)".to_string(),
        ));
    }
    Ok(path)
}

/// Validate user input text (prompts, system messages).
/// Limits length to prevent DoS attacks.
fn validate_user_input(input: &str) -> Result<&str> {
    if input.len() > MAX_INPUT_LENGTH {
        return Err(Error::Validation(format!(
            "Input exceeds maximum length of {} characters",
            MAX_INPUT_LENGTH
        )));
    }
    Ok(input)
}

/// Configuration for the LLM CLI client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCliConfig {
    /// Path to the LLM CLI binary (default: "llm")
    pub binary_path: PathBuf,
    /// Default model to use
    pub default_model: Option<String>,
    /// Default embedding model
    pub default_embedding_model: Option<String>,
    /// Database path for embeddings
    pub database_path: Option<PathBuf>,
}

impl Default for LlmCliConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("llm"),
            default_model: None,
            default_embedding_model: None,
            database_path: None,
        }
    }
}

/// Configuration for embedding operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model to use
    pub model: Option<String>,
    /// Collection name for storing embeddings
    pub collection: Option<String>,
    /// Database path
    pub database: Option<PathBuf>,
    /// Store metadata with embeddings
    pub store_metadata: bool,
}

/// Configuration for clustering operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Embedding model for clustering
    pub model: Option<String>,
    /// Output format (json, csv, etc.)
    pub format: Option<String>,
    /// Include summary in output
    pub summary: bool,
}

/// Configuration for RAG operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Collection to search
    pub collection: String,
    /// Number of results to retrieve
    pub top_k: usize,
    /// Model for generation
    pub model: Option<String>,
    /// System prompt for RAG
    pub system_prompt: Option<String>,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            collection: "default".to_string(),
            top_k: 5,
            model: None,
            system_prompt: None,
        }
    }
}

/// Result from an LLM prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResult {
    /// The generated response
    pub response: String,
    /// Model used
    pub model: String,
    /// Tokens used (if available)
    pub tokens_used: Option<u64>,
}

/// Result from embedding operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Text that was embedded
    pub text: String,
    /// Model used
    pub model: String,
}

/// Result from clustering operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    /// Cluster assignments
    pub clusters: Vec<ClusterAssignment>,
    /// Cluster summaries (if requested)
    pub summaries: Option<Vec<String>>,
}

/// Individual cluster assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAssignment {
    /// Document ID or index
    pub id: String,
    /// Assigned cluster
    pub cluster: usize,
    /// Similarity score within cluster
    pub score: Option<f64>,
}

/// Result from similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Matching documents
    pub matches: Vec<SimilarityMatch>,
}

/// Individual similarity match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    /// Document content
    pub content: String,
    /// Similarity score
    pub score: f64,
    /// Document ID
    pub id: Option<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// LLM CLI Client for multi-model orchestration
#[derive(Debug, Clone)]
pub struct LlmCliClient {
    config: LlmCliConfig,
}

impl LlmCliClient {
    /// Create a new LLM CLI client with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: LlmCliConfig::default(),
        })
    }

    /// Create a new LLM CLI client with custom configuration
    pub fn with_config(config: LlmCliConfig) -> Self {
        Self { config }
    }

    /// Check if the LLM CLI is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.binary_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let output = Command::new(&self.config.binary_path)
            .arg("models")
            .arg("list")
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().map(String::from).collect())
    }

    /// Execute a prompt with the LLM CLI
    pub fn prompt(&self, text: &str, model: Option<&str>) -> Result<PromptResult> {
        // Validate inputs
        let validated_text = validate_user_input(text)?;
        let validated_model = match model {
            Some(m) => Some(validate_model_name(m)?),
            None => None,
        };

        let mut cmd = Command::new(&self.config.binary_path);

        if let Some(m) = validated_model.or(self.config.default_model.as_deref()) {
            cmd.arg("-m").arg(m);
        }

        cmd.arg(validated_text);

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        Ok(PromptResult {
            response: String::from_utf8_lossy(&output.stdout).to_string(),
            model: validated_model
                .or(self.config.default_model.as_deref())
                .unwrap_or("default")
                .to_string(),
            tokens_used: None,
        })
    }

    /// Execute a prompt with system message
    pub fn prompt_with_system(
        &self,
        text: &str,
        system: &str,
        model: Option<&str>,
    ) -> Result<PromptResult> {
        // Validate inputs
        let validated_text = validate_user_input(text)?;
        let validated_system = validate_user_input(system)?;
        let validated_model = match model {
            Some(m) => Some(validate_model_name(m)?),
            None => None,
        };

        let mut cmd = Command::new(&self.config.binary_path);

        if let Some(m) = validated_model.or(self.config.default_model.as_deref()) {
            cmd.arg("-m").arg(m);
        }

        cmd.arg("-s").arg(validated_system);
        cmd.arg(validated_text);

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        Ok(PromptResult {
            response: String::from_utf8_lossy(&output.stdout).to_string(),
            model: validated_model
                .or(self.config.default_model.as_deref())
                .unwrap_or("default")
                .to_string(),
            tokens_used: None,
        })
    }

    /// Generate embeddings for text
    pub fn embed(&self, text: &str, config: Option<&EmbeddingConfig>) -> Result<EmbeddingResult> {
        // Validate inputs
        let validated_text = validate_user_input(text)?;

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("embed");

        // Validate and add optional config
        if let Some(cfg) = config {
            if let Some(ref m) = cfg.model {
                let validated = validate_model_name(m)?;
                cmd.arg("-m").arg(validated);
            }
            if let Some(ref c) = cfg.collection {
                let validated = validate_collection_name(c)?;
                cmd.arg("-c").arg(validated);
            }
            if let Some(ref db) = cfg.database {
                let db_str = db.to_string_lossy();
                let validated = validate_db_path(&db_str)?;
                cmd.arg("-d").arg(validated);
            }
        }

        cmd.arg(validated_text);

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let embedding: Vec<f32> = serde_json::from_str(&stdout).map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Failed to parse embedding response: {}. Response: {}",
                    e, stdout
                )
            ))
        })?;

        Ok(EmbeddingResult {
            embedding,
            text: text.to_string(),
            model: config
                .and_then(|c| c.model.clone())
                .or_else(|| self.config.default_embedding_model.clone())
                .unwrap_or_else(|| "default".to_string()),
        })
    }

    /// Cluster documents
    pub fn cluster(
        &self,
        input: &str,
        num_clusters: usize,
        config: Option<&ClusterConfig>,
    ) -> Result<ClusterResult> {
        // Validate inputs
        let validated_input = validate_user_input(input)?;

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("cluster");
        cmd.arg("-n").arg(num_clusters.to_string());

        if let Some(cfg) = config {
            if let Some(ref m) = cfg.model {
                let validated = validate_model_name(m)?;
                cmd.arg("-m").arg(validated);
            }
            if cfg.summary {
                cmd.arg("--summary");
            }
        }

        cmd.arg(validated_input);

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        // Parse output - simplified for now
        Ok(ClusterResult {
            clusters: vec![],
            summaries: None,
        })
    }

    /// Perform similarity search
    pub fn similar(&self, query: &str, collection: &str, top_k: usize) -> Result<SimilarityResult> {
        // Validate inputs
        let validated_query = validate_user_input(query)?;
        let validated_collection = validate_collection_name(collection)?;

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("similar");
        cmd.arg("-c").arg(validated_collection);
        cmd.arg("-n").arg(top_k.to_string());
        cmd.arg(validated_query);

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        // Parse output - simplified for now
        Ok(SimilarityResult { matches: vec![] })
    }

    /// Execute a RAG pipeline
    pub fn rag(&self, query: &str, config: &RagConfig) -> Result<PromptResult> {
        // First, get similar documents
        let similar = self.similar(query, &config.collection, config.top_k)?;

        // Build context from matches
        let context: String = similar
            .matches
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        // Build RAG prompt
        let rag_prompt = format!(
            "Based on the following context, answer the question.\n\n\
             Context:\n{}\n\n\
             Question: {}",
            context, query
        );

        // Execute with optional system prompt
        if let Some(ref system) = config.system_prompt {
            self.prompt_with_system(&rag_prompt, system, config.model.as_deref())
        } else {
            self.prompt(&rag_prompt, config.model.as_deref())
        }
    }

    /// Execute a prompt using a template
    pub fn prompt_with_template(
        &self,
        template: &str,
        variables: &[(&str, &str)],
        model: Option<&str>,
    ) -> Result<PromptResult> {
        // Validate template name
        let validated_template = validate_template_name(template)?;
        let validated_model = match model {
            Some(m) => Some(validate_model_name(m)?),
            None => None,
        };

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("-t").arg(validated_template);

        if let Some(m) = validated_model.or(self.config.default_model.as_deref()) {
            cmd.arg("-m").arg(m);
        }

        // Add template variables as -p key value pairs
        for (key, value) in variables {
            // Validate variable names (alphanumeric + underscores)
            if !key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            {
                return Err(Error::Validation(format!(
                    "Template variable name contains invalid characters: '{}'",
                    key
                )));
            }
            // Validate variable values
            let validated_value = validate_user_input(value)?;
            cmd.arg("-p").arg(*key).arg(validated_value);
        }

        let output = cmd.output().map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Io(std::io::Error::other(
                String::from_utf8_lossy(&output.stderr).to_string(),
            )));
        }

        Ok(PromptResult {
            response: String::from_utf8_lossy(&output.stdout).to_string(),
            model: validated_model
                .or(self.config.default_model.as_deref())
                .unwrap_or("default")
                .to_string(),
            tokens_used: None,
        })
    }
}

impl Default for LlmCliClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default LlmCliClient")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model_name_valid() {
        assert!(validate_model_name("gpt-4o-mini").is_ok());
        assert!(validate_model_name("claude-sonnet-4").is_ok());
        assert!(validate_model_name("sentence-transformers/all-MiniLM-L6-v2").is_ok());
        assert!(validate_model_name("model:latest").is_ok());
    }

    #[test]
    fn test_validate_model_name_invalid() {
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("model; rm -rf /").is_err());
        assert!(validate_model_name("model$(whoami)").is_err());
        assert!(validate_model_name("../../../etc/passwd").is_err());
    }

    #[test]
    fn test_validate_collection_name_valid() {
        assert!(validate_collection_name("my-collection").is_ok());
        assert!(validate_collection_name("collection_123").is_ok());
    }

    #[test]
    fn test_validate_collection_name_invalid() {
        assert!(validate_collection_name("").is_err());
        assert!(validate_collection_name("collection/path").is_err());
        assert!(validate_collection_name("col; drop table").is_err());
    }

    #[test]
    fn test_validate_db_path_valid() {
        assert!(validate_db_path("/home/user/.llm/db.sqlite").is_ok());
        assert!(validate_db_path("./data/embeddings.db").is_ok());
    }

    #[test]
    fn test_validate_db_path_invalid() {
        assert!(validate_db_path("").is_err());
        assert!(validate_db_path("/path/../../../etc/passwd").is_err());
        assert!(validate_db_path("/path$(whoami)/db").is_err());
        assert!(validate_db_path("/path`id`/db").is_err());
        assert!(validate_db_path("/path;rm -rf/db").is_err());
    }

    #[test]
    fn test_format_dangerous_char() {
        assert_eq!(format_dangerous_char('\n'), "\\n");
        assert_eq!(format_dangerous_char('\r'), "\\r");
        assert_eq!(format_dangerous_char('\0'), "\\0");
        assert_eq!(format_dangerous_char('$'), "$");
    }

    #[test]
    fn test_client_creation() {
        let client = LlmCliClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_config_default() {
        let config = LlmCliConfig::default();
        assert_eq!(config.binary_path, PathBuf::from("llm"));
        assert!(config.default_model.is_none());
    }
}
