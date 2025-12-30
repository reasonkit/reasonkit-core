//! MCP Tools for Cross-Encoder Reranking
//!
//! Exposes the Reranker module as MCP tools for agent consumption.
//!
//! ## Available Tools
//!
//! - `rerank` - Rerank a list of candidates given a query
//! - `rerank_config` - Get or set reranker configuration
//!
//! ## Research Foundation
//!
//! Based on: "Cross-Encoders for Document Reranking" (arXiv:2010.06467)
//! Target: MRR@10 > 0.40 on MS MARCO, latency < 200ms for top-20

use super::tools::{Tool, ToolResult, ToolResultContent};
#[cfg(feature = "memory")]
use reasonkit_mem::retrieval::{
    RerankStats, RerankedResult, Reranker, RerankerCandidate, RerankerConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Rerank tool handler
pub struct RerankToolHandler {
    /// Reranker instance
    reranker: Arc<RwLock<Reranker>>,
}

/// Input for rerank tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankInput {
    /// Search query
    pub query: String,
    /// Candidates to rerank
    pub candidates: Vec<CandidateInput>,
    /// Number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    10
}

/// Candidate input format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInput {
    /// Unique ID (optional, will be generated if not provided)
    pub id: Option<String>,
    /// Document/chunk text
    pub text: String,
    /// Original score (optional)
    #[serde(default)]
    pub score: f32,
}

/// Output for rerank tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankOutput {
    /// Reranked results
    pub results: Vec<RerankedResult>,
    /// Statistics
    pub stats: RerankStats,
}

impl RerankToolHandler {
    /// Create a new rerank tool handler
    pub fn new() -> Self {
        Self {
            reranker: Arc::new(RwLock::new(Reranker::new(RerankerConfig::default()))),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RerankerConfig) -> Self {
        Self {
            reranker: Arc::new(RwLock::new(Reranker::new(config))),
        }
    }

    /// Get tool definitions for MCP registration
    pub fn tool_definitions() -> Vec<Tool> {
        vec![
            Tool {
                name: "rerank".to_string(),
                description: Some(
                    "Rerank search results using cross-encoder scoring for improved precision. \
                     Takes a query and list of candidates, returns candidates sorted by relevance. \
                     Target latency: <200ms for 20 candidates."
                        .to_string(),
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to score candidates against"
                        },
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Optional unique identifier"
                                    },
                                    "text": {
                                        "type": "string",
                                        "description": "Document or chunk text to score"
                                    },
                                    "score": {
                                        "type": "number",
                                        "description": "Original retrieval score (for tracking)"
                                    }
                                },
                                "required": ["text"]
                            },
                            "description": "List of candidates to rerank"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query", "candidates"]
                }),
                server_id: None,
                server_name: None,
            },
            Tool {
                name: "rerank_config".to_string(),
                description: Some(
                    "Get or update reranker configuration. Returns current config if no \
                     parameters provided, otherwise updates the config."
                        .to_string(),
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "preset": {
                            "type": "string",
                            "enum": ["default", "fast", "quality"],
                            "description": "Use a preset configuration"
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Minimum score threshold (0-1)"
                        }
                    }
                }),
                server_id: None,
                server_name: None,
            },
        ]
    }

    /// Handle rerank tool call
    pub async fn handle_rerank(&self, input: RerankInput) -> ToolResult {
        let start = std::time::Instant::now();

        // Convert input candidates to internal format
        let candidates: Vec<RerankerCandidate> = input
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| RerankerCandidate {
                id: c
                    .id
                    .as_ref()
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .unwrap_or_else(Uuid::new_v4),
                text: c.text.clone(),
                original_score: c.score,
                original_rank: i,
            })
            .collect();

        // Perform reranking
        let reranker = self.reranker.read().await;
        match reranker
            .rerank(&input.query, &candidates, input.top_k)
            .await
        {
            Ok(results) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let stats = RerankStats::from_results(&results, latency_ms);

                let output = RerankOutput { results, stats };

                ToolResult {
                    content: vec![ToolResultContent::Text {
                        text: serde_json::to_string_pretty(&output).unwrap_or_else(|e| {
                            format!("{{\"error\": \"Serialization failed: {}\"}}", e)
                        }),
                    }],
                    is_error: Some(false),
                }
            }
            Err(e) => ToolResult {
                content: vec![ToolResultContent::Text {
                    text: json!({
                        "error": e.to_string()
                    })
                    .to_string(),
                }],
                is_error: Some(true),
            },
        }
    }

    /// Handle rerank_config tool call
    pub async fn handle_config(&self, input: Value) -> ToolResult {
        let reranker = self.reranker.read().await;
        let current_config = reranker.config();

        // If preset is specified, describe what it would do
        if let Some(preset) = input.get("preset").and_then(|v| v.as_str()) {
            let config = match preset {
                "fast" => RerankerConfig::fast(),
                "quality" => RerankerConfig::quality(),
                _ => RerankerConfig::default(),
            };

            let response = json!({
                "current": {
                    "model_id": current_config.model_id,
                    "max_length": current_config.max_length,
                    "batch_size": current_config.batch_size,
                    "use_gpu": current_config.use_gpu,
                    "score_threshold": current_config.score_threshold
                },
                "requested_preset": preset,
                "preset_config": {
                    "model_id": config.model_id,
                    "max_length": config.max_length,
                    "batch_size": config.batch_size,
                    "use_gpu": config.use_gpu
                },
                "note": "Config changes require server restart to take effect"
            });

            ToolResult {
                content: vec![ToolResultContent::Text {
                    text: serde_json::to_string_pretty(&response).unwrap(),
                }],
                is_error: Some(false),
            }
        } else {
            // Just return current config
            let response = json!({
                "config": {
                    "model_id": current_config.model_id,
                    "max_length": current_config.max_length,
                    "batch_size": current_config.batch_size,
                    "use_gpu": current_config.use_gpu,
                    "score_threshold": current_config.score_threshold,
                    "enable_cache": current_config.enable_cache
                },
                "available_presets": ["default", "fast", "quality"]
            });

            ToolResult {
                content: vec![ToolResultContent::Text {
                    text: serde_json::to_string_pretty(&response).unwrap(),
                }],
                is_error: Some(false),
            }
        }
    }

    /// Handle any rerank tool call
    pub async fn handle_tool(&self, name: &str, arguments: Value) -> ToolResult {
        match name {
            "rerank" => match serde_json::from_value::<RerankInput>(arguments) {
                Ok(input) => self.handle_rerank(input).await,
                Err(e) => ToolResult {
                    content: vec![ToolResultContent::Text {
                        text: json!({
                            "error": format!("Invalid input: {}", e)
                        })
                        .to_string(),
                    }],
                    is_error: Some(true),
                },
            },
            "rerank_config" => self.handle_config(arguments).await,
            _ => ToolResult {
                content: vec![ToolResultContent::Text {
                    text: json!({
                        "error": format!("Unknown tool: {}", name)
                    })
                    .to_string(),
                }],
                is_error: Some(true),
            },
        }
    }
}

impl Default for RerankToolHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_definitions() {
        let tools = RerankToolHandler::tool_definitions();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "rerank");
        assert_eq!(tools[1].name, "rerank_config");
    }

    #[tokio::test]
    async fn test_rerank_tool() {
        let handler = RerankToolHandler::new();

        let input = RerankInput {
            query: "machine learning".to_string(),
            candidates: vec![
                CandidateInput {
                    id: None,
                    text: "Machine learning is a subset of AI.".to_string(),
                    score: 0.8,
                },
                CandidateInput {
                    id: None,
                    text: "The weather is sunny today.".to_string(),
                    score: 0.7,
                },
            ],
            top_k: 2,
        };

        let result = handler.handle_rerank(input).await;
        assert_eq!(result.is_error, Some(false));
    }

    #[tokio::test]
    async fn test_config_tool() {
        let handler = RerankToolHandler::new();

        let result = handler.handle_config(json!({})).await;
        assert_eq!(result.is_error, Some(false));

        let result = handler.handle_config(json!({"preset": "fast"})).await;
        assert_eq!(result.is_error, Some(false));
    }
}
