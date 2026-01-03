//! # GLM-4.6 Type Definitions
//!
//! Complete type system for GLM-4.6 API integration.
//! Optimized for structured output and agent coordination.

use serde::{Deserialize, Serialize};
// use std::collections::HashMap;

/// Chat message for GLM-4.6 API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    /// Optional tool calls for agentic coordination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional tool response for tool execution results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a tool message (for tool execution results)
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Message role in conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature (0.0 to 1.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Response format for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Optional tools for agentic coordination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Top probability sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Enable streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Response format for structured output
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    /// Structured output with JSON schema
    JsonSchema {
        name: String,
        schema: serde_json::Value,
    },
    /// Custom structured format
    Structured,
}

/// Tool definition for agentic coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String, // "function"
    pub function: ToolFunction,
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call made by the assistant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // "function"
    pub function: ToolFunctionCall,
}

/// Tool function call details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool choice setting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    #[serde(untagged)]
    Specific(ToolCallChoice),
}

/// Specific tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChoice {
    pub r#type: String, // "function"
    pub function: ToolFunctionRef,
}

/// Tool function reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionRef {
    pub name: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: TokenUsage,
    /// System fingerprint (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Single choice in chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Reason for finishing the generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Convenience properties for usage
impl TokenUsage {
    /// Alias for prompt_tokens (input tokens)
    pub fn input_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Alias for completion_tokens (output tokens)
    pub fn output_tokens(&self) -> usize {
        self.completion_tokens
    }
}

/// Stream response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// Stream choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: ChatMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Message delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy {
        /// Round-trip latency
        latency: std::time::Duration,
    },
    Error {
        /// HTTP status code if available
        status: Option<u16>,
        /// Error message
        message: String,
    },
}

/// Usage statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub total_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
    pub session_start: std::time::SystemTime,
}

impl UsageStats {
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens + self.total_output_tokens
    }

    pub fn average_tokens_per_request(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_tokens() as f64 / self.total_requests as f64
        }
    }

    pub fn cost_per_1k_tokens(&self) -> f64 {
        let total_tokens = self.total_tokens();
        if total_tokens == 0 {
            0.0
        } else {
            (self.total_cost / total_tokens as f64) * 1000.0
        }
    }
}

/// Circuit breaker state for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,
    Open {
        opens_at: std::time::SystemTime,
        reset_after: std::time::Duration,
    },
    HalfOpen {
        probation_requests: u32,
        max_probation: u32,
    },
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to trigger circuit opening
    pub failure_threshold: u32,
    /// Success threshold to close circuit
    pub success_threshold: u32,
    /// Timeout for half-open state
    pub timeout: std::time::Duration,
    /// Reset timeout for open circuit
    pub reset_timeout: std::time::Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: std::time::Duration::from_secs(60),
            reset_timeout: std::time::Duration::from_secs(300),
        }
    }
}

/// API request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: usize,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

impl APIRequest {
    pub fn from_chat_request(
        request: &ChatRequest,
        config: &crate::glm46::client::GLM46Config,
    ) -> Self {
        Self {
            model: config.model.clone(),
            messages: request.messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
            response_format: request.response_format.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            stop: request.stop.clone(),
        }
    }

    pub fn from_chat_request_stream(
        request: &ChatRequest,
        config: &crate::glm46::client::GLM46Config,
    ) -> Self {
        let mut req = Self::from_chat_request(request, config);
        req.stream = true;
        req
    }
}

/// API response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: TokenUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<APIError>,
}

/// API error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl APIResponse {
    pub fn into_chat_response(self) -> ChatResponse {
        ChatResponse {
            id: self.id,
            object: self.object,
            created: self.created,
            model: self.model,
            choices: self.choices,
            usage: self.usage,
            system_fingerprint: self.system_fingerprint,
        }
    }
}

/// Local Ollama configuration for fallback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server URL
    pub url: String,
    /// Model name
    pub model: String,
    /// Request timeout
    pub timeout: std::time::Duration,
    /// Whether to use local fallback
    pub enabled: bool,
    /// Minimum retry count before fallback
    pub fallback_threshold: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "glm4".to_string()),
            timeout: std::time::Duration::from_secs(120),
            enabled: std::env::var("OLLAMA_ENABLED").unwrap_or_else(|_| "true".to_string())
                == "true",
            fallback_threshold: 3,
        }
    }
}

/// Error types for GLM-4.6 client
#[derive(Debug, thiserror::Error)]
pub enum GLM46Error {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("API error: {message}")]
    API {
        message: String,
        code: Option<String>,
    },

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Timeout error: operation took longer than {0:?}")]
    Timeout(std::time::Duration),

    #[error("Circuit breaker is open: {reason}")]
    CircuitOpen { reason: String },

    #[error("Ollama error: {0}")]
    Ollama(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Context window exceeded: {used} > {limit}")]
    ContextExceeded { used: usize, limit: usize },

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Rate limited: retry after {retry_after:?}")]
    RateLimited { retry_after: std::time::Duration },
}

/// Custom result type for GLM-4.6 operations
pub type GLM46Result<T> = Result<T, GLM46Error>;
