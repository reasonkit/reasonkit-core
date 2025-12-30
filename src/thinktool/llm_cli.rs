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

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

use crate::error::{Error, Result};

/// LLM CLI client for interacting with Simon Willison's `llm` tool
pub struct LlmCliClient {
    /// Path to the `llm` binary (defaults to "llm" in PATH)
    llm_binary: String,
    /// Default model to use
    default_model: Option<String>,
    /// Logs database path (from `llm logs path`)
    logs_db_path: Option<PathBuf>,
}

impl LlmCliClient {
    /// Create a new LLM CLI client
    pub fn new() -> Result<Self> {
        // Check if `llm` is available
        let output = Command::new("llm").arg("--version").output().map_err(|e| {
            Error::Config(format!(
                "llm CLI not found: {}. Install with: uv tool install llm",
                e
            ))
        })?;

        if !output.status.success() {
            return Err(Error::Config("llm CLI not working properly".to_string()));
        }

        // Try to get logs path
        let logs_db_path = Command::new("llm")
            .arg("logs")
            .arg("path")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| PathBuf::from(s.trim()))
                } else {
                    None
                }
            });

        Ok(Self {
            llm_binary: "llm".to_string(),
            default_model: None,
            logs_db_path,
        })
    }

    /// Create with a specific model
    pub fn with_model(model: &str) -> Result<Self> {
        let mut client = Self::new()?;
        client.default_model = Some(model.to_string());
        Ok(client)
    }

    /// Execute a prompt with optional model
    pub async fn prompt(&self, prompt: &str, model: Option<&str>) -> Result<LlmCliResponse> {
        let model = model.or(self.default_model.as_deref());
        let mut cmd = Command::new(&self.llm_binary);

        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }

        cmd.arg(prompt);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm command failed: {}", stderr)));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(LlmCliResponse {
            text: text.trim().to_string(),
            model: model.map(|s| s.to_string()),
        })
    }

    /// Execute a prompt with system message
    pub async fn prompt_with_system(
        &self,
        prompt: &str,
        system: &str,
        model: Option<&str>,
    ) -> Result<LlmCliResponse> {
        let model = model.or(self.default_model.as_deref());
        let mut cmd = Command::new(&self.llm_binary);

        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }

        cmd.arg("-s").arg(system).arg(prompt);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm command failed: {}", stderr)));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(LlmCliResponse {
            text: text.trim().to_string(),
            model: model.map(|s| s.to_string()),
        })
    }

    /// Execute a prompt with a template
    pub async fn prompt_with_template(
        &self,
        prompt: &str,
        template: &str,
        model: Option<&str>,
    ) -> Result<LlmCliResponse> {
        let model = model.or(self.default_model.as_deref());
        let mut cmd = Command::new(&self.llm_binary);

        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }

        cmd.arg("-t").arg(template).arg(prompt);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm command failed: {}", stderr)));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(LlmCliResponse {
            text: text.trim().to_string(),
            model: model.map(|s| s.to_string()),
        })
    }

    /// Generate embeddings for text
    pub async fn embed(&self, text: &str, model: Option<&str>) -> Result<Vec<f32>> {
        let model = model.unwrap_or("sentence-transformers/all-MiniLM-L6-v2");
        let mut cmd = Command::new(&self.llm_binary);

        cmd.arg("embed").arg("-m").arg(model).arg("-c").arg(text);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm embed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm embed failed: {}", stderr)));
        }

        // Parse JSON array of floats
        let text = String::from_utf8_lossy(&output.stdout);
        let embeddings: Vec<f32> = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("Failed to parse embeddings JSON: {}", e)))?;

        Ok(embeddings)
    }

    /// Embed multiple items and store in database
    pub async fn embed_multi(
        &self,
        collection: &str,
        db_path: &str,
        sql: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        let model = model.unwrap_or("sentence-transformers/all-MiniLM-L6-v2");
        let mut cmd = Command::new(&self.llm_binary);

        cmd.arg("embed-multi")
            .arg(collection)
            .arg("-d")
            .arg(db_path)
            .arg("-m")
            .arg(model)
            .arg("--store");

        if let Some(s) = sql {
            cmd.arg("--sql").arg(s);
        }

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm embed-multi: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!(
                "llm embed-multi failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    /// Cluster items in a database
    pub async fn cluster(
        &self,
        collection: &str,
        num_clusters: usize,
        db_path: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<ClusterResult>> {
        let db_path = db_path.unwrap_or(
            self.logs_db_path
                .as_ref()
                .and_then(|p| p.to_str())
                .ok_or_else(|| Error::Config("Database path required".to_string()))?,
        );

        let model = model.unwrap_or("gpt-4o-mini");
        let mut cmd = Command::new(&self.llm_binary);

        cmd.arg("cluster")
            .arg(collection)
            .arg(num_clusters.to_string())
            .arg("-d")
            .arg(db_path)
            .arg("-m")
            .arg(model)
            .arg("--summary");

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm cluster: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm cluster failed: {}", stderr)));
        }

        // Parse JSON output
        let text = String::from_utf8_lossy(&output.stdout);
        let clusters: Vec<ClusterResult> = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("Failed to parse cluster JSON: {}", e)))?;

        Ok(clusters)
    }

    /// Query using RAG tool
    pub async fn rag_query(
        &self,
        query: &str,
        collection: Option<&str>,
        model: Option<&str>,
    ) -> Result<LlmCliResponse> {
        let model = model.or(self.default_model.as_deref());
        let mut cmd = Command::new(&self.llm_binary);

        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }

        if let Some(c) = collection {
            cmd.arg("-T").arg(format!("RAG(\"{}\")", c));
        } else {
            cmd.arg("-T").arg("RAG");
        }

        cmd.arg(query);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm RAG query: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!("llm RAG query failed: {}", stderr)));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(LlmCliResponse {
            text: text.trim().to_string(),
            model: model.map(|s| s.to_string()),
        })
    }

    /// Query SQLite database using llm-tools-sqlite
    pub async fn sqlite_query(
        &self,
        query: &str,
        db_path: &str,
        model: Option<&str>,
    ) -> Result<LlmCliResponse> {
        let model = model.or(self.default_model.as_deref());
        let mut cmd = Command::new(&self.llm_binary);

        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }

        cmd.arg("-T")
            .arg(format!("SQLite(\"{}\")", db_path))
            .arg(query);

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm SQLite query: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!(
                "llm SQLite query failed: {}",
                stderr
            )));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(LlmCliResponse {
            text: text.trim().to_string(),
            model: model.map(|s| s.to_string()),
        })
    }

    /// Get logs database path
    pub fn logs_db_path(&self) -> Option<&PathBuf> {
        self.logs_db_path.as_ref()
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let mut cmd = Command::new(&self.llm_binary);
        cmd.arg("models").arg("list");

        let output = cmd
            .output()
            .map_err(|e| Error::Network(format!("Failed to execute llm models list: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!(
                "llm models list failed: {}",
                stderr
            )));
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let models: Vec<String> = text
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    None
                } else {
                    Some(line.to_string())
                }
            })
            .collect();

        Ok(models)
    }
}

/// Response from LLM CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCliResponse {
    /// Response text
    pub text: String,
    /// Model used (if known)
    pub model: Option<String>,
}

/// Cluster result from `llm cluster`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    /// Cluster ID
    pub id: usize,
    /// Cluster summary
    pub summary: String,
    /// Items in cluster
    pub items: Vec<ClusterItem>,
}

/// Item in a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterItem {
    /// Item ID
    pub id: String,
    /// Item content
    pub content: String,
    /// Similarity score
    pub similarity: Option<f64>,
}

/// Configuration for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model to use
    pub model: String,
    /// Database path
    pub db_path: Option<PathBuf>,
    /// Collection name
    pub collection: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            db_path: None,
            collection: None,
        }
    }
}

/// Configuration for clustering operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Number of clusters
    pub num_clusters: usize,
    /// Model to use for summarization
    pub model: String,
    /// Database path
    pub db_path: Option<PathBuf>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            num_clusters: 5,
            model: "gpt-4o-mini".to_string(),
            db_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_cli_client_new() {
        // This test will fail if `llm` is not installed, which is expected
        // In CI/CD, we should skip this test or mock it
        let result = LlmCliClient::new();
        // Just verify it doesn't panic - actual functionality requires `llm` CLI
        if result.is_err() {
            // Expected if `llm` is not installed
            println!("llm CLI not available (expected in some environments)");
        }
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model, "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.num_clusters, 5);
        assert_eq!(config.model, "gpt-4o-mini");
    }
}
