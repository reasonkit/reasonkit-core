//! Embedding module for ReasonKit Core
//!
//! Provides text embedding functionality using various backends:
//! - API-based: OpenAI, Anthropic, Cohere, local servers
//! - Local ONNX: BGE-M3, E5, etc.
//! - Hybrid: Dense + Sparse (SPLADE)

pub mod cache;
#[cfg(feature = "local-embeddings")]
pub mod local;

use crate::{Chunk, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub use cache::EmbeddingCache;

/// Embedding vector type
pub type EmbeddingVector = Vec<f32>;

/// Sparse embedding (for BM25/SPLADE)
pub type SparseEmbedding = HashMap<u32, f32>;

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier
    pub model: String,
    /// Embedding dimension
    pub dimension: usize,
    /// API endpoint (for API-based models)
    pub api_endpoint: Option<String>,
    /// API key environment variable
    pub api_key_env: Option<String>,
    /// Maximum batch size
    pub batch_size: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds (0 = no expiration)
    pub cache_ttl_secs: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
            api_endpoint: Some("https://api.openai.com/v1/embeddings".to_string()),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            batch_size: 100,
            normalize: true,
            timeout_secs: 30,
            enable_cache: true,
            cache_ttl_secs: 86400, // 24 hours
        }
    }
}

impl EmbeddingConfig {
    /// Create a config for local BGE-M3 model
    #[cfg(feature = "local-embeddings")]
    pub fn bge_m3() -> Self {
        Self {
            model: "BAAI/bge-m3".to_string(),
            dimension: 1024,
            api_endpoint: None,
            api_key_env: None,
            batch_size: 32,
            normalize: true,
            timeout_secs: 60,
            enable_cache: true,
            cache_ttl_secs: 86400,
        }
    }

    /// Create a config for local E5-small model
    #[cfg(feature = "local-embeddings")]
    pub fn e5_small() -> Self {
        Self {
            model: "intfloat/e5-small-v2".to_string(),
            dimension: 384,
            api_endpoint: None,
            api_key_env: None,
            batch_size: 64,
            normalize: true,
            timeout_secs: 60,
            enable_cache: true,
            cache_ttl_secs: 86400,
        }
    }
}

/// Embedding result for a single text
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Dense embedding vector
    pub dense: Option<EmbeddingVector>,
    /// Sparse embedding (token -> weight)
    pub sparse: Option<SparseEmbedding>,
    /// Token count
    pub token_count: usize,
}

/// Trait for embedding providers
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Embed a single text
    async fn embed(&self, text: &str) -> Result<EmbeddingResult>;

    /// Embed multiple texts (batch)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>>;
}

/// OpenAI-compatible embedding provider
pub struct OpenAIEmbedding {
    config: EmbeddingConfig,
    client: reqwest::Client,
    cache: Option<Arc<EmbeddingCache>>,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding provider
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::embedding(format!("Failed to create HTTP client: {}", e)))?;

        let cache = if config.enable_cache {
            Some(Arc::new(EmbeddingCache::new(
                10000, // max_entries
                config.cache_ttl_secs,
            )))
        } else {
            None
        };

        Ok(Self {
            config,
            client,
            cache,
        })
    }

    /// Create with default OpenAI config
    pub fn openai() -> Result<Self> {
        Self::new(EmbeddingConfig::default())
    }

    /// Create with custom cache
    pub fn with_cache(mut self, cache: Arc<EmbeddingCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Get API key from environment
    fn get_api_key(&self) -> Result<String> {
        let env_var = self
            .config
            .api_key_env
            .as_deref()
            .unwrap_or("OPENAI_API_KEY");
        std::env::var(env_var).map_err(|_| {
            Error::embedding(format!(
                "API key not found in environment variable: {}",
                env_var
            ))
        })
    }

    /// Generate cache key for a text
    fn cache_key(&self, text: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.config.model.as_bytes());
        hasher.update(b":");
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAIEmbedding {
    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            let key = self.cache_key(text);
            if let Some(cached) = cache.get(&key) {
                return Ok(EmbeddingResult {
                    dense: Some(cached),
                    sparse: None,
                    token_count: text.split_whitespace().count(),
                });
            }
        }

        // Fallback to batch embedding
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::embedding("Empty response from embedding API"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check cache for all texts
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        if let Some(ref cache) = self.cache {
            for (i, text) in texts.iter().enumerate() {
                let key = self.cache_key(text);
                if let Some(cached) = cache.get(&key) {
                    results.push(EmbeddingResult {
                        dense: Some(cached),
                        sparse: None,
                        token_count: text.split_whitespace().count(),
                    });
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }
        } else {
            uncached_indices.extend(0..texts.len());
            uncached_texts.extend(texts.iter());
        }

        // If all cached, return early
        if uncached_texts.is_empty() {
            return Ok(results);
        }

        let api_key = self.get_api_key()?;
        let endpoint = self
            .config
            .api_endpoint
            .as_deref()
            .unwrap_or("https://api.openai.com/v1/embeddings");

        // Build request
        let request_body = serde_json::json!({
            "model": self.config.model,
            "input": uncached_texts,
        });

        let response = self
            .client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::embedding(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::embedding(format!("API error {}: {}", status, body)));
        }

        // Parse response
        let response_body: OpenAIEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::embedding(format!("Failed to parse response: {}", e)))?;

        // Convert to EmbeddingResult and cache
        let mut new_results = Vec::with_capacity(uncached_texts.len());
        for (i, data) in response_body.data.iter().enumerate() {
            let embedding = if self.config.normalize {
                normalize_vector(&data.embedding)
            } else {
                data.embedding.clone()
            };

            // Cache the embedding
            if let Some(ref cache) = self.cache {
                let key = self.cache_key(uncached_texts[i]);
                cache.put(key, embedding.clone());
            }

            new_results.push(EmbeddingResult {
                dense: Some(embedding),
                sparse: None,
                token_count: response_body.usage.prompt_tokens / uncached_texts.len(),
            });
        }

        // Merge cached and new results in correct order
        if self.cache.is_some() {
            let mut final_results = Vec::with_capacity(texts.len());
            let mut new_idx = 0;
            for i in 0..texts.len() {
                if uncached_indices.contains(&i) {
                    final_results.push(new_results[new_idx].clone());
                    new_idx += 1;
                } else {
                    final_results.push(results.remove(0));
                }
            }
            Ok(final_results)
        } else {
            Ok(new_results)
        }
    }
}

/// OpenAI API response structure
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Embedding pipeline for processing documents
pub struct EmbeddingPipeline {
    provider: Arc<dyn EmbeddingProvider>,
    batch_size: usize,
}

impl EmbeddingPipeline {
    /// Create a new embedding pipeline
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            provider,
            batch_size: 100,
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Embed a list of chunks
    pub async fn embed_chunks(&self, chunks: &[Chunk]) -> Result<Vec<EmbeddingResult>> {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

        // Process in batches
        let mut all_results = Vec::with_capacity(chunks.len());

        for batch in texts.chunks(self.batch_size) {
            let results = self.provider.embed_batch(batch).await?;
            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Embed a single text
    pub async fn embed_text(&self, text: &str) -> Result<EmbeddingVector> {
        let result = self.provider.embed(text).await?;
        result
            .dense
            .ok_or_else(|| Error::embedding("No dense embedding returned"))
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    /// Get the provider
    pub fn provider(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.provider
    }
}

/// Normalize a vector to unit length
pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        v.iter().map(|x| x / magnitude).collect()
    } else {
        v.to_vec()
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimension, 1536);
        assert!(config.api_endpoint.is_some());
        assert!(config.enable_cache);
    }

    #[test]
    #[cfg(feature = "local-embeddings")]
    fn test_embedding_config_local() {
        let config = EmbeddingConfig::bge_m3();
        assert_eq!(config.model, "BAAI/bge-m3");
        assert_eq!(config.dimension, 1024);
        assert!(config.api_endpoint.is_none());

        let config = EmbeddingConfig::e5_small();
        assert_eq!(config.model, "intfloat/e5-small-v2");
        assert_eq!(config.dimension, 384);
    }
}
