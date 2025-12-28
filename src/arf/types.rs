//! Core type definitions for the ARF system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for reasoning sessions
pub type SessionId = String;

/// Unique identifier for reasoning steps
pub type StepId = String;

/// Configuration for the ARF system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArfConfig {
    pub version: String,
    pub runtime: RuntimeConfig,
    pub engine: EngineConfig,
    pub plugins: PluginConfig,
    pub logging: LoggingConfig,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_concurrent_sessions: usize,
    pub session_timeout_seconds: u64,
    pub worker_threads: usize,
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub max_steps_per_session: usize,
    pub step_timeout_seconds: u64,
    pub validation_enabled: bool,
    pub cognitive_load_monitoring: bool,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub enabled: bool,
    pub plugin_directory: String,
    pub max_plugins: usize,
    pub hot_reload: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file_output: Option<String>,
}

/// Reasoning session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSession {
    pub id: SessionId,
    pub problem_statement: String,
    pub status: SessionStatus,
    pub current_step: usize,
    pub total_steps: usize,
    pub steps: Vec<ReasoningStep>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Session status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    #[serde(rename = "initialized")]
    Initialized,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "paused")]
    Paused,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub id: StepId,
    pub step_number: usize,
    pub name: String,
    pub instruction: String,
    pub cognitive_stance: String,
    pub time_allocation: String,
    pub status: StepStatus,
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
    pub validation_result: Option<ValidationResult>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Step status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "skipped")]
    Skipped,
}

/// Validation result for step outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub score: f64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub dependencies: Vec<String>,
    pub author: String,
}

/// Cognitive load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadMetrics {
    pub intrinsic_load: f64,
    pub extraneous_load: f64,
    pub germane_load: f64,
    pub total_load: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub session_id: SessionId,
    pub total_duration_ms: u64,
    pub average_step_duration_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub token_efficiency: f64,
    pub validation_success_rate: f64,
}

/// Search and discovery results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub query: String,
    pub results: Vec<SearchItem>,
    pub total_matches: usize,
    pub search_duration_ms: u64,
}

/// Individual search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchItem {
    pub path: String,
    pub line_number: Option<usize>,
    pub content: String,
    pub context: Vec<String>,
    pub relevance_score: f64,
}

/// MCP (Model Context Protocol) request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub method: String,
    pub params: serde_json::Value,
    pub id: Option<serde_json::Value>,
}

/// MCP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub result: Option<serde_json::Value>,
    pub error: Option<McpError>,
    pub id: Option<serde_json::Value>,
}

/// MCP error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}
