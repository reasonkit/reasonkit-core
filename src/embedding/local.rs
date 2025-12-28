//! Local ONNX-based embedding providers
//!
//! Provides local embedding inference using ONNX Runtime for models like:
//! - BGE-M3 (1024 dims, multilingual)
//! - E5-small-v2 (384 dims, efficient)

use super::{
    normalize_vector, EmbeddingCache, EmbeddingConfig, EmbeddingProvider, EmbeddingResult,
};
use crate::{Error, Result};
use async_trait::async_trait;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Local ONNX embedding provider
pub struct LocalONNXEmbedding {
    config: EmbeddingConfig,
    session: Session,
    tokenizer: Tokenizer,
    cache: Option<Arc<EmbeddingCache>>,
}

impl LocalONNXEmbedding {
    /// Create a new local ONNX embedding provider
    ///
    /// # Arguments
    /// * `model_path` - Path to ONNX model file
    /// * `tokenizer_path` - Path to tokenizer JSON file
    /// * `config` - Embedding configuration
    pub fn new(
        model_path: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
        config: EmbeddingConfig,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let tokenizer_path = tokenizer_path.into();

        // Initialize ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("reasonkit-embedding")
            .build()
            .map_err(|e| Error::embedding(format!("Failed to create ONNX environment: {}", e)))?;

        // Create session with optimizations
        let session = SessionBuilder::new(&environment)
            .map_err(|e| Error::embedding(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .map_err(|e| Error::embedding(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| Error::embedding(format!("Failed to set intra threads: {}", e)))?
            .with_model_from_file(&model_path)
            .map_err(|e| {
                Error::embedding(format!(
                    "Failed to load ONNX model from {:?}: {}",
                    model_path, e
                ))
            })?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            Error::embedding(format!(
                "Failed to load tokenizer from {:?}: {}",
                tokenizer_path, e
            ))
        })?;

        let cache = if config.enable_cache {
            Some(Arc::new(EmbeddingCache::new(10000, config.cache_ttl_secs)))
        } else {
            None
        };

        Ok(Self {
            config,
            session,
            tokenizer,
            cache,
        })
    }

    /// Create BGE-M3 embedding provider
    ///
    /// # Arguments
    /// * `models_dir` - Directory containing ONNX model and tokenizer
    pub fn bge_m3(models_dir: impl Into<PathBuf>) -> Result<Self> {
        let models_dir = models_dir.into();
        Self::new(
            models_dir.join("bge-m3.onnx"),
            models_dir.join("bge-m3-tokenizer.json"),
            EmbeddingConfig::bge_m3(),
        )
    }

    /// Create E5-small embedding provider
    ///
    /// # Arguments
    /// * `models_dir` - Directory containing ONNX model and tokenizer
    pub fn e5_small(models_dir: impl Into<PathBuf>) -> Result<Self> {
        let models_dir = models_dir.into();
        Self::new(
            models_dir.join("e5-small-v2.onnx"),
            models_dir.join("e5-small-v2-tokenizer.json"),
            EmbeddingConfig::e5_small(),
        )
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

    /// Perform ONNX inference
    fn infer(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Tokenize inputs
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("Tokenization failed: {}", e)))?;

        let batch_size = texts.len();
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        // Prepare input tensors (input_ids, attention_mask)
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for (j, &id) in ids.iter().enumerate() {
                input_ids[i * max_len + j] = id as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
            }
        }

        // Create ONNX tensors
        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| Error::embedding(format!("Failed to create input_ids array: {}", e)))?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, max_len), attention_mask).map_err(
                |e| Error::embedding(format!("Failed to create attention_mask array: {}", e)),
            )?;

        // Run inference
        let outputs = self
            .session
            .run(vec![
                Value::from_array(self.session.allocator(), &input_ids_array).map_err(|e| {
                    Error::embedding(format!("Failed to create input_ids tensor: {}", e))
                })?,
                Value::from_array(self.session.allocator(), &attention_mask_array).map_err(
                    |e| Error::embedding(format!("Failed to create attention_mask tensor: {}", e)),
                )?,
            ])
            .map_err(|e| Error::embedding(format!("ONNX inference failed: {}", e)))?;

        // Extract embeddings from output
        let embeddings_tensor = outputs
            .get(0)
            .ok_or_else(|| Error::embedding("No output from ONNX model"))?;

        let embeddings_array = embeddings_tensor
            .try_extract::<f32>()
            .map_err(|e| Error::embedding(format!("Failed to extract embeddings: {}", e)))?
            .view()
            .to_owned();

        // Convert to Vec<Vec<f32>>
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let embedding = embeddings_array.slice(ndarray::s![i, ..]).to_vec();

            let embedding = if self.config.normalize {
                normalize_vector(&embedding)
            } else {
                embedding
            };

            results.push(embedding);
        }

        Ok(results)
    }
}

#[async_trait]
impl EmbeddingProvider for LocalONNXEmbedding {
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

        // Perform inference
        let embeddings = self.infer(&[text])?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::embedding("No embedding returned"))?;

        // Cache the result
        if let Some(ref cache) = self.cache {
            let key = self.cache_key(text);
            cache.put(key, embedding.clone());
        }

        Ok(EmbeddingResult {
            dense: Some(embedding),
            sparse: None,
            token_count: text.split_whitespace().count(),
        })
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

        // Perform batch inference
        let embeddings = self.infer(&uncached_texts)?;

        // Cache and prepare results
        let mut new_results = Vec::with_capacity(uncached_texts.len());
        for (i, embedding) in embeddings.into_iter().enumerate() {
            // Cache the embedding
            if let Some(ref cache) = self.cache {
                let key = self.cache_key(uncached_texts[i]);
                cache.put(key, embedding.clone());
            }

            new_results.push(EmbeddingResult {
                dense: Some(embedding),
                sparse: None,
                token_count: uncached_texts[i].split_whitespace().count(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_provider_creation() {
        // This is a placeholder test - actual testing requires model files
        // In production, you would download models from HuggingFace
        let result = LocalONNXEmbedding::new(
            PathBuf::from("models/bge-m3.onnx"),
            PathBuf::from("models/bge-m3-tokenizer.json"),
            EmbeddingConfig::bge_m3(),
        );

        // Will fail without actual model files, which is expected
        assert!(result.is_err());
    }
}
