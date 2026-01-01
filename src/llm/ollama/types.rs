use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Chat message compatible with Ollama `/api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request payload for Ollama `/api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model identifier (e.g. `deepseek-v3.2:cloud`).
    pub model: String,

    /// Conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Whether to stream partial results.
    ///
    /// This client supports only `stream:false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Optional decoding/sampling parameters.
    ///
    /// Ollama accepts an `options` object with provider-specific keys. We keep it
    /// flexible for forward-compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<BTreeMap<String, Value>>,
}

/// Assistant message inside Ollama chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: String,
}

/// Response payload for Ollama `/api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub message: ChatResponseMessage,

    #[serde(default)]
    pub done: bool,

    /// Capture other fields returned by Ollama without hard-coding schema.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Best-effort error envelope.
///
/// Ollama commonly returns: `{ "error": "..." }`.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{error}")]
pub struct OllamaErrorEnvelope {
    pub error: String,

    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
