//! Protocol Delta MCP Tools
//!
//! MCP tool implementations for Protocol Delta's verification capabilities.
//! These tools expose the ProofLedger immutable citation ledger to high-level
//! agents (Claude, Grok, Gemini) for source verification and drift detection.
//!
//! ## Available Tools
//!
//! - `proof_anchor` - Anchor content to the immutable ledger
//! - `proof_verify` - Verify content against an anchored hash
//! - `proof_lookup` - Look up an anchor by hash
//! - `proof_list_by_url` - List all anchors for a URL
//! - `proof_stats` - Get ledger statistics
//!
//! ## Philosophy
//!
//! > "We do not quote the wind. We quote the stone."
//!
//! Protocol Delta replaces weak URL citations with cryptographically-bound
//! anchors that can detect content drift over time.

use super::tools::{Tool, ToolResult};
use crate::verification::{ProofLedger, ProofLedgerError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Protocol Delta tool handler (uses ProofLedger)
pub struct DeltaToolHandler {
    /// ProofLedger instance
    ledger: Arc<RwLock<ProofLedger>>,
    /// Ledger path (kept for potential future use like backup/export)
    #[allow(dead_code)]
    ledger_path: PathBuf,
}

/// Input for delta_anchor tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaAnchorInput {
    /// Content to anchor
    pub content: String,
    /// Source URL
    pub url: String,
    /// Optional metadata (JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

/// Input for delta_verify tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaVerifyInput {
    /// Original hash from citation
    pub hash: String,
    /// Current content to verify
    pub content: String,
}

/// Input for delta_lookup tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaLookupInput {
    /// Hash to look up
    pub hash: String,
}

/// Input for delta_list_by_url tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaListByUrlInput {
    /// URL to search for
    pub url: String,
}

/// Output for delta_anchor tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaAnchorOutput {
    /// SHA-256 hash (citation ID)
    pub hash: String,
    /// Short hash for display (first 8 chars)
    pub short_hash: String,
    /// Source URL
    pub url: String,
    /// Timestamp (RFC3339)
    pub timestamp: String,
    /// Citation format for reports
    pub citation: String,
}

/// Output for delta_verify tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaVerifyOutput {
    /// Whether verification passed
    pub verified: bool,
    /// Original hash
    pub original_hash: String,
    /// Current content hash
    pub current_hash: String,
    /// Human-readable status
    pub status: String,
    /// Recommendation for the agent
    pub recommendation: String,
}

impl DeltaToolHandler {
    /// Create a new Protocol Delta tool handler
    ///
    /// # Arguments
    ///
    /// * `ledger_path` - Path to the SQLite ledger database
    pub fn new<P: AsRef<Path>>(ledger_path: P) -> Result<Self, ProofLedgerError> {
        let path = ledger_path.as_ref().to_path_buf();
        let ledger = ProofLedger::new(&path)?;

        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Self {
            ledger: Arc::new(RwLock::new(ledger)),
            ledger_path: path,
        })
    }

    /// Create an in-memory handler (for testing)
    pub fn in_memory() -> Result<Self, ProofLedgerError> {
        let ledger = ProofLedger::in_memory()?;

        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Self {
            ledger: Arc::new(RwLock::new(ledger)),
            ledger_path: PathBuf::from(":memory:"),
        })
    }

    /// Get tool definitions for MCP registration
    pub fn tool_definitions() -> Vec<Tool> {
        vec![
            Tool::with_schema(
                "delta_anchor",
                "Anchor content to Protocol Delta's immutable citation ledger. \
                Returns a SHA-256 hash that can be used as a verifiable citation. \
                Use this BEFORE making claims based on external sources.",
                json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The exact content/text to anchor (will be hashed)"
                        },
                        "url": {
                            "type": "string",
                            "description": "Source URL where the content was retrieved from"
                        },
                        "metadata": {
                            "type": "string",
                            "description": "Optional JSON metadata (source type, confidence, etc.)"
                        }
                    },
                    "required": ["content", "url"]
                }),
            ),
            Tool::with_schema(
                "delta_verify",
                "Verify that content matches an anchored citation. \
                Use this to detect if a source has changed (content drift) \
                since it was originally cited.",
                json!({
                    "type": "object",
                    "properties": {
                        "hash": {
                            "type": "string",
                            "description": "The original SHA-256 hash from the citation"
                        },
                        "content": {
                            "type": "string",
                            "description": "The current content to verify against the anchor"
                        }
                    },
                    "required": ["hash", "content"]
                }),
            ),
            Tool::with_schema(
                "delta_lookup",
                "Look up an anchor by its hash. Returns the original URL, \
                timestamp, and content snippet for a given citation hash.",
                json!({
                    "type": "object",
                    "properties": {
                        "hash": {
                            "type": "string",
                            "description": "The SHA-256 hash to look up"
                        }
                    },
                    "required": ["hash"]
                }),
            ),
            Tool::with_schema(
                "delta_list_by_url",
                "List all anchored citations from a specific URL. \
                Useful for finding historical citations from a source.",
                json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to search for"
                        }
                    },
                    "required": ["url"]
                }),
            ),
            Tool::with_schema(
                "delta_stats",
                "Get statistics about the Protocol Delta ledger. \
                Returns total anchors, ledger path, and status.",
                json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            ),
        ]
    }

    /// Handle a delta_anchor tool call
    pub async fn handle_anchor(
        &self,
        args: &HashMap<String, Value>,
    ) -> Result<ToolResult, ProofLedgerError> {
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProofLedgerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Missing 'content' argument",
                ))
            })?;

        let url = args.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
            ProofLedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Missing 'url' argument",
            ))
        })?;

        let metadata = args
            .get("metadata")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let ledger = self.ledger.read().await;
        let hash = ledger.anchor(content, url, metadata)?;

        let output = DeltaAnchorOutput {
            hash: hash.clone(),
            short_hash: hash[..8].to_string(),
            url: url.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            citation: format!(
                "[sha256:{}...] (Anchored {}) → {}",
                &hash[..8],
                chrono::Utc::now().format("%Y-%m-%d"),
                url
            ),
        };

        let json_output = serde_json::to_string_pretty(&output).unwrap_or_else(|_| hash.clone());

        Ok(ToolResult::text(format!(
            "ANCHORED: Content successfully bound to immutable ledger.\n\n\
            Citation ID: {}\n\
            Short Hash: {}\n\
            Source: {}\n\n\
            Use this citation format in reports:\n\
            {}\n\n\
            Raw JSON:\n{}",
            output.hash, output.short_hash, output.url, output.citation, json_output
        )))
    }

    /// Handle a delta_verify tool call
    pub async fn handle_verify(
        &self,
        args: &HashMap<String, Value>,
    ) -> Result<ToolResult, ProofLedgerError> {
        let hash = args.get("hash").and_then(|v| v.as_str()).ok_or_else(|| {
            ProofLedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Missing 'hash' argument",
            ))
        })?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProofLedgerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Missing 'content' argument",
                ))
            })?;

        let ledger = self.ledger.read().await;
        let result = ledger.verify(hash, content)?;

        let output = DeltaVerifyOutput {
            verified: result.verified,
            original_hash: result.original_hash.clone(),
            current_hash: result.current_hash.clone(),
            status: if result.verified {
                "VERIFIED".to_string()
            } else {
                "DRIFT_DETECTED".to_string()
            },
            recommendation: if result.verified {
                "Citation is valid. Content matches the original anchor.".to_string()
            } else {
                format!(
                    "WARNING: Content has changed since anchoring. \
                    Original hash: {}..., Current hash: {}... \
                    Consider re-anchoring or flagging this citation as outdated.",
                    &result.original_hash[..8],
                    &result.current_hash[..8]
                )
            },
        };

        let json_output = serde_json::to_string_pretty(&output)
            .unwrap_or_else(|_| format!("verified: {}", result.verified));

        let status_icon = if result.verified { "✓" } else { "⚠" };

        Ok(ToolResult::text(format!(
            "{} {}\n\n\
            Original Hash: {}...\n\
            Current Hash:  {}...\n\
            Match: {}\n\n\
            Recommendation: {}\n\n\
            Raw JSON:\n{}",
            status_icon,
            output.status,
            &output.original_hash[..8],
            &output.current_hash[..8],
            output.verified,
            output.recommendation,
            json_output
        )))
    }

    /// Handle a delta_lookup tool call
    pub async fn handle_lookup(
        &self,
        args: &HashMap<String, Value>,
    ) -> Result<ToolResult, ProofLedgerError> {
        let hash = args.get("hash").and_then(|v| v.as_str()).ok_or_else(|| {
            ProofLedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Missing 'hash' argument",
            ))
        })?;

        let ledger = self.ledger.read().await;
        let anchor = ledger.get_anchor(hash)?;

        let json_output = serde_json::to_string_pretty(&anchor)
            .unwrap_or_else(|_| format!("hash: {}", anchor.hash));

        Ok(ToolResult::text(format!(
            "ANCHOR FOUND\n\n\
            Hash: {}\n\
            URL: {}\n\
            Timestamp: {}\n\
            Snippet: {}...\n\n\
            Raw JSON:\n{}",
            anchor.hash,
            anchor.url,
            anchor.timestamp.to_rfc3339(),
            &anchor.content_snippet[..anchor.content_snippet.len().min(100)],
            json_output
        )))
    }

    /// Handle a delta_list_by_url tool call
    pub async fn handle_list_by_url(
        &self,
        args: &HashMap<String, Value>,
    ) -> Result<ToolResult, ProofLedgerError> {
        let url = args.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
            ProofLedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Missing 'url' argument",
            ))
        })?;

        let ledger = self.ledger.read().await;
        let anchors = ledger.list_by_url(url)?;

        if anchors.is_empty() {
            return Ok(ToolResult::text(format!(
                "No anchors found for URL: {}\n\n\
                This URL has no citations in the ledger. \
                Use delta_anchor to create the first citation.",
                url
            )));
        }

        let mut output = format!("ANCHORS FOR URL: {}\nTotal: {}\n\n", url, anchors.len());

        for (i, anchor) in anchors.iter().enumerate() {
            output.push_str(&format!(
                "{}. [{}...] - {}\n   Snippet: {}...\n\n",
                i + 1,
                &anchor.hash[..8],
                anchor.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                &anchor.content_snippet[..anchor.content_snippet.len().min(60)]
            ));
        }

        let json_output =
            serde_json::to_string_pretty(&anchors).unwrap_or_else(|_| "[]".to_string());

        output.push_str(&format!("\nRaw JSON:\n{}", json_output));

        Ok(ToolResult::text(output))
    }

    /// Handle a delta_stats tool call
    pub async fn handle_stats(&self) -> Result<ToolResult, ProofLedgerError> {
        let ledger = self.ledger.read().await;
        let count = ledger.count()?;
        let path = ledger.ledger_path();

        let stats = json!({
            "total_anchors": count,
            "ledger_path": path.to_string_lossy(),
            "status": "operational",
            "protocol_version": "delta_v2"
        });

        let json_output = serde_json::to_string_pretty(&stats).unwrap_or_else(|_| "{}".to_string());

        Ok(ToolResult::text(format!(
            "PROTOCOL DELTA LEDGER STATUS\n\n\
            Total Anchors: {}\n\
            Ledger Path: {}\n\
            Status: Operational\n\
            Protocol Version: Delta V2 (Amber)\n\n\
            Raw JSON:\n{}",
            count,
            path.display(),
            json_output
        )))
    }

    /// Dispatch a tool call to the appropriate handler
    pub async fn handle_tool(
        &self,
        name: &str,
        args: &HashMap<String, Value>,
    ) -> Result<ToolResult, ProofLedgerError> {
        match name {
            "delta_anchor" => self.handle_anchor(args).await,
            "delta_verify" => self.handle_verify(args).await,
            "delta_lookup" => self.handle_lookup(args).await,
            "delta_list_by_url" => self.handle_list_by_url(args).await,
            "delta_stats" => self.handle_stats().await,
            _ => Ok(ToolResult::error(format!("Unknown tool: {}", name))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::ToolResultContent;

    #[tokio::test]
    async fn test_delta_anchor() {
        let handler = DeltaToolHandler::in_memory().unwrap();

        let mut args = HashMap::new();
        args.insert("content".to_string(), json!("Test content for anchoring"));
        args.insert("url".to_string(), json!("https://example.com/test"));

        let result = handler.handle_anchor(&args).await.unwrap();

        assert!(result.is_error.is_none() || !result.is_error.unwrap());
        if let ToolResultContent::Text { text } = &result.content[0] {
            assert!(text.contains("ANCHORED"));
            assert!(text.contains("sha256"));
        }
    }

    #[tokio::test]
    async fn test_delta_verify_success() {
        let handler = DeltaToolHandler::in_memory().unwrap();

        // First anchor
        let mut anchor_args = HashMap::new();
        anchor_args.insert("content".to_string(), json!("Immutable content"));
        anchor_args.insert("url".to_string(), json!("https://example.com"));

        let anchor_result = handler.handle_anchor(&anchor_args).await.unwrap();

        // Extract hash from result
        let hash = if let ToolResultContent::Text { text } = &anchor_result.content[0] {
            // Parse hash from output
            text.lines()
                .find(|l: &&str| l.starts_with("Citation ID:"))
                .and_then(|l: &str| l.split(':').nth(1))
                .map(|s: &str| s.trim().to_string())
                .unwrap()
        } else {
            panic!("Expected text content");
        };

        // Now verify
        let mut verify_args = HashMap::new();
        verify_args.insert("hash".to_string(), json!(hash));
        verify_args.insert("content".to_string(), json!("Immutable content"));

        let result = handler.handle_verify(&verify_args).await.unwrap();

        if let ToolResultContent::Text { text } = &result.content[0] {
            assert!(text.contains("VERIFIED"));
            assert!(text.contains("Match: true"));
        }
    }

    #[tokio::test]
    async fn test_delta_verify_drift() {
        let handler = DeltaToolHandler::in_memory().unwrap();

        // Anchor original
        let mut anchor_args = HashMap::new();
        anchor_args.insert("content".to_string(), json!("Original content"));
        anchor_args.insert("url".to_string(), json!("https://example.com"));

        let anchor_result = handler.handle_anchor(&anchor_args).await.unwrap();

        let hash = if let ToolResultContent::Text { text } = &anchor_result.content[0] {
            text.lines()
                .find(|l: &&str| l.starts_with("Citation ID:"))
                .and_then(|l: &str| l.split(':').nth(1))
                .map(|s: &str| s.trim().to_string())
                .unwrap()
        } else {
            panic!("Expected text content");
        };

        // Verify with different content
        let mut verify_args = HashMap::new();
        verify_args.insert("hash".to_string(), json!(hash));
        verify_args.insert("content".to_string(), json!("Modified content"));

        let result = handler.handle_verify(&verify_args).await.unwrap();

        if let ToolResultContent::Text { text } = &result.content[0] {
            assert!(text.contains("DRIFT_DETECTED"));
            assert!(text.contains("WARNING"));
        }
    }

    #[tokio::test]
    async fn test_delta_stats() {
        let handler = DeltaToolHandler::in_memory().unwrap();

        let result = handler.handle_stats().await.unwrap();

        if let ToolResultContent::Text { text } = &result.content[0] {
            assert!(text.contains("PROTOCOL DELTA LEDGER STATUS"));
            assert!(text.contains("Total Anchors:"));
            assert!(text.contains("Delta V2"));
        }
    }

    #[tokio::test]
    async fn test_tool_definitions() {
        let tools = DeltaToolHandler::tool_definitions();

        assert_eq!(tools.len(), 5);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"delta_anchor"));
        assert!(names.contains(&"delta_verify"));
        assert!(names.contains(&"delta_lookup"));
        assert!(names.contains(&"delta_list_by_url"));
        assert!(names.contains(&"delta_stats"));
    }
}
