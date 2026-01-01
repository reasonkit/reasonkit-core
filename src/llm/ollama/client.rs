use crate::llm::ollama::types::{ChatRequest, ChatResponse, OllamaErrorEnvelope};
use reqwest::{header, StatusCode};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
    timeout: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaClientError {
    #[error("http error: {0}")]
    Transport(#[from] reqwest::Error),

    #[error("unexpected status {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },

    #[error("ollama error: {0}")]
    Ollama(#[from] OllamaErrorEnvelope),

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("streaming not supported by this client; set stream=false")]
    StreamingNotSupported,
}

impl OllamaClient {
    /// `base_url` examples:
    /// - `http://localhost:11434`
    /// - `http://127.0.0.1:11434`
    pub fn new(base_url: impl Into<String>) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("reasonkit-ollama-client/", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(60),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }

    /// Call `/api/chat` with `stream:false`.
    pub async fn chat(&self, mut req: ChatRequest) -> Result<ChatResponse, OllamaClientError> {
        if req.stream.is_none() {
            req.stream = Some(false);
        }
        if req.stream != Some(false) {
            return Err(OllamaClientError::StreamingNotSupported);
        }

        let resp = self
            .http
            .post(self.chat_url())
            .header(header::ACCEPT, "application/json")
            .json(&req)
            .timeout(self.timeout)
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;

        if !status.is_success() {
            if let Ok(err_env) = serde_json::from_str::<OllamaErrorEnvelope>(&body) {
                return Err(OllamaClientError::Ollama(err_env));
            }
            return Err(OllamaClientError::HttpStatus { status, body });
        }

        serde_json::from_str::<ChatResponse>(&body)
            .map_err(|e| OllamaClientError::InvalidResponse(format!("{e}; body={body}")))
    }
}
