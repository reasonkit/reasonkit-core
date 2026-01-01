//! Stubbed M2 connector.
use crate::error::Error;
use crate::m2::types::{
    CompositeConstraints, Evidence, ExecutionMetrics, InterleavedProtocol, M2Config, ProtocolInput,
    ProtocolOutput as M2ProtocolOutput,
};
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use tracing::{debug, error, info, instrument};

#[derive(Debug)]
pub struct M2Connector {
    client: Client,
    config: M2Config,
}

impl M2Connector {
    pub fn new(config: M2Config) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    #[instrument(skip(self, protocol, input))]
    pub async fn execute_interleaved_thinking(
        &self,
        protocol: &InterleavedProtocol,
        _constraints: &CompositeConstraints,
        input: &ProtocolInput,
    ) -> Result<M2Result, Error> {
        let endpoint = &self.config.endpoint;

        if endpoint.contains("ollama") || endpoint.contains("localhost") {
            return self.execute_via_ollama(protocol, input).await.map_err(|e| {
                error!("Ollama execution failed: {}", e);
                Error::M2ExecutionError(format!("Ollama execution failed: {}", e))
            });
        }

        // Dummy implementation for non-Ollama endpoints (until we have a real external API)
        info!("Executing M2 connector stub for endpoint: {}", endpoint);
        Ok(M2Result {
            output: M2ProtocolOutput {
                result: serde_json::Value::Null.to_string(),
                evidence: vec![],
                confidence: 0.0,
            },
            metrics: ExecutionMetrics::default(),
        })
    }

    async fn execute_via_ollama(
        &self,
        protocol: &InterleavedProtocol,
        input: &ProtocolInput,
    ) -> Result<M2Result> {
        let model = "minimax-m2.1:cloud"; // Or derive from config
        let prompt = format!(
            "Execute the following protocol:\nName: {}\nDescription: {}\n\nInput: {}",
            protocol.name, protocol.description, input
        );

        let body = json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        });

        debug!("Sending request to Ollama at {}", self.config.endpoint);

        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&body)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to send request to Ollama endpoint: {}",
                    self.config.endpoint
                )
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read error body: {}>", e));
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                status,
                error_body
            ));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse Ollama response as JSON")?;

        // Safely extract the response text, defaulting to empty string if not present
        let response_text = response_json
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Basic parsing of response to M2Result
        // In a real implementation, we might parse specific sections or evidence
        Ok(M2Result {
            output: M2ProtocolOutput {
                result: response_text.clone(),
                evidence: vec![Evidence {
                    content: "Generated via Ollama".to_string(),
                    source: "minimax-m2.1".to_string(),
                    confidence: 0.8, // Placeholder confidence
                }],
                confidence: 0.8,
            },
            metrics: ExecutionMetrics {
                duration_ms: 0,                  // We could measure this
                token_usage: Default::default(), // Parse from response if available
                ..Default::default()
            },
        })
    }
}

#[derive(Debug)]
pub struct M2Result {
    pub output: M2ProtocolOutput,
    pub metrics: ExecutionMetrics,
}
