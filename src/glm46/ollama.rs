//! # Local Ollama Fallback
//!
//! Local deployment compatibility for GLM-4.6 when cloud API fails.
//! Uses ollama for local inference with graceful degradation.

// use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
// use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::glm46::types::{
    ChatMessage, ChatRequest, ChatResponse, Choice, FinishReason, GLM46Error, GLM46Result,
    OllamaConfig, TokenUsage,
};

/// Ollama client for local fallback
#[derive(Debug, Clone)]
pub struct OllamaClient {
    config: OllamaConfig,
    http_client: Client,
}

impl OllamaClient {
    /// Create new Ollama client
    pub fn new(config: OllamaConfig) -> Self {
        let http_client = Client::builder()
            .timeout(config.timeout)
            .user_agent("reasonkit-glm46/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            http_client,
        }
    }

    /// Check if Ollama is available
    pub async fn is_available(&self) -> bool {
        let response = self
            .http_client
            .get(format!("{}/api/version", self.config.url))
            .send()
            .await;

        response.is_ok_and(|r| r.status().is_success())
    }

    /// Execute chat completion with fallback
    pub async fn chat_completion(&self, request: ChatRequest) -> GLM46Result<ChatResponse> {
        if !self.config.enabled {
            return Err(GLM46Error::Config("Ollama fallback disabled".to_string()));
        }

        debug!(
            "Executing fallback request via Ollama: {}",
            self.config.model
        );

        let ollama_request = OllamaRequest::from_chat_request(request, &self.config);

        let response = timeout(self.config.timeout, self.send_request(&ollama_request))
            .await
            .map_err(|_| GLM46Error::Timeout(self.config.timeout))?
            .map_err(GLM46Error::Network)?;

        self.parse_response(response).await
    }

    /// Stream chat completion via Ollama
    pub async fn stream_chat_completion(
        &self,
        request: ChatRequest,
    ) -> GLM46Result<tokio::sync::mpsc::UnboundedReceiver<crate::glm46::types::StreamChunk>> {
        if !self.config.enabled {
            return Err(GLM46Error::Config("Ollama fallback disabled".to_string()));
        }

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let client = self.clone();

        tokio::spawn(async move {
            let ollama_request = OllamaRequest::from_chat_request(request, &client.config);

            match client.send_stream_request(ollama_request, tx).await {
                Ok(_) => debug!("Ollama stream completed successfully"),
                Err(e) => warn!("Ollama stream error: {:?}", e),
            }
        });

        Ok(rx)
    }

    /// Send request to Ollama API
    async fn send_request(
        &self,
        request: &OllamaRequest,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}/api/chat", self.config.url);

        debug!("Sending request to Ollama at: {}", url);

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        Ok(response)
    }

    /// Send streaming request to Ollama
    async fn send_stream_request(
        &self,
        request: OllamaRequest,
        tx: tokio::sync::mpsc::UnboundedSender<crate::glm46::types::StreamChunk>,
    ) -> GLM46Result<()> {
        let url = format!("{}/api/chat", self.config.url);

        let request_builder = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json");

        // Add streaming parameters
        let mut stream_request = request.clone();
        stream_request.stream = true;

        let response = request_builder.json(&stream_request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(GLM46Error::API {
                message: format!("Ollama API error {}: {}", status, body),
                code: Some(status.as_u16().to_string()),
            });
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        use futures::stream::StreamExt;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete JSON objects
            while let Some(end_idx) = buffer.find('}') {
                if let Some(start_idx) = buffer.rfind('{') {
                    if start_idx < end_idx {
                        let json_str = buffer[start_idx..=end_idx].to_string();
                        buffer = buffer[end_idx + 1..].to_string();

                        match serde_json::from_str::<OllamaResponse>(&json_str) {
                            Ok(ollama_response) => {
                                let chunk = crate::glm46::types::StreamChunk {
                                    id: "ollama-stream".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    model: self.config.model.clone(),
                                    choices: vec![crate::glm46::types::StreamChoice {
                                        index: 0,
                                        delta: crate::glm46::types::ChatMessageDelta {
                                            role: if ollama_response.message.role == "assistant" {
                                                Some(crate::glm46::types::MessageRole::Assistant)
                                            } else {
                                                None
                                            },
                                            content: Some(ollama_response.message.content.clone()),
                                            tool_calls: None,
                                        },
                                        finish_reason: if ollama_response.done {
                                            Some(crate::glm46::types::FinishReason::Stop)
                                        } else {
                                            None
                                        },
                                    }],
                                    usage: ollama_response.done.then(|| {
                                        crate::glm46::types::TokenUsage {
                                            prompt_tokens: ollama_response
                                                .prompt_eval_count
                                                .unwrap_or(0),
                                            completion_tokens: ollama_response
                                                .eval_count
                                                .unwrap_or(0),
                                            total_tokens: ollama_response
                                                .prompt_eval_count
                                                .unwrap_or(0)
                                                + ollama_response.eval_count.unwrap_or(0),
                                        }
                                    }),
                                };

                                if tx.send(chunk).is_err() {
                                    break; // Channel closed
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to parse Ollama stream chunk: {:?}\nData: {}",
                                    e, json_str
                                );
                            }
                        }
                    }
                }
                // If no valid JSON found, clear buffer and continue
                buffer.clear();
            }
        }

        Ok(())
    }

    /// Parse Ollama response
    async fn parse_response(&self, response: reqwest::Response) -> GLM46Result<ChatResponse> {
        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(GLM46Error::API {
                message: format!("Ollama error {}: {}", status, body),
                code: Some(status.as_u16().to_string()),
            });
        }

        let ollama_response: OllamaResponse =
            serde_json::from_str(&body).map_err(GLM46Error::Json)?;

        debug!(
            "Ollama response: done={}, content_length={}",
            ollama_response.done,
            ollama_response.message.content.len()
        );

        Ok(ollama_response.into_chat_response())
    }
}

/// Ollama API request format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub format: Option<String>, // "json" for structured output
    pub stream: bool,
    pub options: Option<OllamaOptions>,
}

impl OllamaRequest {
    pub fn from_chat_request(request: ChatRequest, config: &OllamaConfig) -> Self {
        Self {
            model: config.model.clone(),
            messages: request.messages.into_iter().map(Into::into).collect(),
            format: if request.response_format.is_some() {
                Some("json".to_string())
            } else {
                None
            },
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(request.temperature),
                num_predict: Some(request.max_tokens as u32),
                top_p: request.top_p,
                stop: request.stop,
            }),
        }
    }
}

/// Ollama message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for OllamaMessage {
    fn from(msg: ChatMessage) -> Self {
        Self {
            role: match msg.role {
                crate::glm46::types::MessageRole::System => "system".to_string(),
                crate::glm46::types::MessageRole::User => "user".to_string(),
                crate::glm46::types::MessageRole::Assistant => "assistant".to_string(),
                crate::glm46::types::MessageRole::Tool => "tool".to_string(),
            },
            content: msg.content,
        }
    }
}

/// Ollama generation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub num_predict: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
}

/// Ollama API response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponse {
    pub message: OllamaMessage,
    pub done: bool,
    pub prompt_eval_count: Option<usize>,
    pub eval_count: Option<usize>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
}

impl OllamaResponse {
    pub fn into_chat_response(self) -> ChatResponse {
        ChatResponse {
            id: format!("ollama-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: "ollama-local".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: crate::glm46::types::MessageRole::Assistant,
                    content: self.message.content,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: if self.done {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                },
            }],
            usage: TokenUsage {
                prompt_tokens: self.prompt_eval_count.unwrap_or(0),
                completion_tokens: self.eval_count.unwrap_or(0),
                total_tokens: self.prompt_eval_count.unwrap_or(0) + self.eval_count.unwrap_or(0),
            },
            system_fingerprint: Some(format!(
                "{:?}+{:?}",
                self.total_duration, self.load_duration
            )),
        }
    }
}

/// Fallback manager for GLM-4.6 client
#[derive(Debug)]
pub struct FallbackManager {
    ollama_client: OllamaClient,
    enabled: bool,
    failure_threshold: u32,
    consecutive_failures: std::sync::atomic::AtomicU32,
}

impl FallbackManager {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            ollama_client: OllamaClient::new(config.clone()),
            enabled: config.enabled,
            failure_threshold: config.fallback_threshold,
            consecutive_failures: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Check if fallback should be used
    pub async fn should_use_fallback(&self) -> bool {
        self.enabled
            && self.ollama_client.is_available().await
            && self
                .consecutive_failures
                .load(std::sync::atomic::Ordering::Relaxed)
                >= self.failure_threshold
    }

    /// Record API failure
    pub fn record_failure(&self) {
        let failures = self
            .consecutive_failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        warn!("GLM-4.6 API failure #{}", failures);

        if failures >= self.failure_threshold {
            info!(
                "Activating Ollama fallback after {} consecutive failures",
                failures
            );
        }
    }

    /// Record API success
    pub fn record_success(&self) {
        self.consecutive_failures
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Execute fallback request
    pub async fn execute_fallback(&self, request: ChatRequest) -> GLM46Result<ChatResponse> {
        if !self.should_use_fallback().await {
            return Err(GLM46Error::Config(
                "Ollama fallback not available".to_string(),
            ));
        }

        info!("Executing request via Ollama fallback");
        self.ollama_client.chat_completion(request).await
    }

    /// Get current status
    pub async fn get_status(&self) -> FallbackStatus {
        FallbackStatus {
            enabled: self.enabled,
            ollama_available: self.ollama_client.is_available().await,
            consecutive_failures: self
                .consecutive_failures
                .load(std::sync::atomic::Ordering::Relaxed),
            failure_threshold: self.failure_threshold,
            active: self.should_use_fallback().await,
        }
    }
}

/// Fallback status information
#[derive(Debug, Clone)]
pub struct FallbackStatus {
    pub enabled: bool,
    pub ollama_available: bool,
    pub consecutive_failures: u32,
    pub failure_threshold: u32,
    pub active: bool,
}
