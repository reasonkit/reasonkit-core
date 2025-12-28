//! Cross-Encoder Reranking Module
//!
//! Implements document reranking using cross-encoder models for improved
//! retrieval precision.
//!
//! ## Research Foundation
//!
//! Based on: "Cross-Encoders for Document Reranking"
//! - **Paper:** <https://arxiv.org/abs/2010.06467>
//! - **Claim:** Cross-encoders achieve SOTA reranking quality
//! - **Target:** MRR@10 > 0.40 on MS MARCO, latency < 200ms for top-20
//!
//! ## Architecture
//!
//! ```text
//! Query + Document → Cross-Encoder → Relevance Score
//!
//! Unlike bi-encoders (separate embeddings):
//!   Query → Encoder → embedding₁  ─┐
//!   Doc   → Encoder → embedding₂  ─┴→ cosine_similarity
//!
//! Cross-encoders process the pair together:
//!   [CLS] Query [SEP] Document [SEP] → Transformer → Score
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::retrieval::{Reranker, RerankerConfig, HybridResult};
//!
//! // Create reranker with configuration
//! let reranker = Reranker::new(RerankerConfig::default());
//!
//! // Rerank results from hybrid search
//! let reranked = reranker.rerank(query, &results, top_k).await?;
//! ```

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for the cross-encoder reranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Model identifier (e.g., "ms-marco-MiniLM-L-12-v2")
    pub model_id: String,

    /// Maximum sequence length for cross-encoder input
    pub max_length: usize,

    /// Batch size for inference
    pub batch_size: usize,

    /// Whether to use GPU for inference
    pub use_gpu: bool,

    /// Minimum score threshold (filter out below this)
    pub score_threshold: Option<f32>,

    /// Enable caching of (query, doc) pairs
    pub enable_cache: bool,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-MiniLM-L-12-v2".to_string(),
            max_length: 512,
            batch_size: 16,
            use_gpu: false,
            score_threshold: None,
            enable_cache: true,
        }
    }
}

impl RerankerConfig {
    /// Create config for fast reranking (smaller model)
    pub fn fast() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 256,
            batch_size: 32,
            use_gpu: false,
            score_threshold: None,
            enable_cache: true,
        }
    }

    /// Create config for high-quality reranking (larger model)
    pub fn quality() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-electra-base".to_string(),
            max_length: 512,
            batch_size: 8,
            use_gpu: true,
            score_threshold: None,
            enable_cache: true,
        }
    }
}

/// A candidate document for reranking
#[derive(Debug, Clone)]
pub struct RerankerCandidate {
    /// Unique identifier
    pub id: Uuid,

    /// Document/chunk text
    pub text: String,

    /// Original retrieval score
    pub original_score: f32,

    /// Original rank position
    pub original_rank: usize,
}

/// Reranked result with cross-encoder score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedResult {
    /// Unique identifier
    pub id: Uuid,

    /// Document/chunk text
    pub text: String,

    /// Cross-encoder relevance score (0-1, higher = more relevant)
    pub rerank_score: f32,

    /// Original retrieval score
    pub original_score: f32,

    /// Original rank position
    pub original_rank: usize,

    /// New rank after reranking
    pub new_rank: usize,

    /// Rank improvement (positive = moved up)
    pub rank_delta: i32,
}

/// Cross-encoder model backend trait
#[async_trait]
pub trait CrossEncoderBackend: Send + Sync {
    /// Score a batch of (query, document) pairs
    async fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Get the model identifier
    fn model_id(&self) -> &str;

    /// Get maximum sequence length
    fn max_length(&self) -> usize;
}

/// Placeholder cross-encoder for testing
///
/// Uses a simple heuristic: term overlap between query and document
pub struct HeuristicCrossEncoder {
    model_id: String,
    max_length: usize,
}

impl Default for HeuristicCrossEncoder {
    fn default() -> Self {
        Self {
            model_id: "heuristic-overlap".to_string(),
            max_length: 512,
        }
    }
}

impl HeuristicCrossEncoder {
    /// Create a new heuristic cross-encoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute term overlap score between query and document
    fn term_overlap_score(query: &str, document: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<String> = query_lower
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let doc_lower = document.to_lowercase();

        if query_terms.is_empty() {
            return 0.0;
        }

        let matches = query_terms
            .iter()
            .filter(|term| doc_lower.contains(term.as_str()))
            .count();

        matches as f32 / query_terms.len() as f32
    }
}

#[async_trait]
impl CrossEncoderBackend for HeuristicCrossEncoder {
    async fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|doc| Self::term_overlap_score(query, doc))
            .collect())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Cross-encoder reranker for improving retrieval precision
pub struct Reranker {
    config: RerankerConfig,
    backend: Arc<dyn CrossEncoderBackend>,
}

impl Reranker {
    /// Create a new reranker with heuristic backend (for testing)
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            backend: Arc::new(HeuristicCrossEncoder::default()),
        }
    }

    /// Create a reranker with a custom backend
    pub fn with_backend(config: RerankerConfig, backend: Arc<dyn CrossEncoderBackend>) -> Self {
        Self { config, backend }
    }

    /// Get the configuration
    pub fn config(&self) -> &RerankerConfig {
        &self.config
    }

    /// Rerank a list of candidates
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `candidates` - List of candidates to rerank
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Reranked results sorted by cross-encoder score
    pub async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        top_k: usize,
    ) -> Result<Vec<RerankedResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Extract texts for scoring
        let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();

        // Score all (query, document) pairs
        let scores = self.backend.score_pairs(query, &texts).await?;

        // Create results with scores
        let mut results: Vec<RerankedResult> = candidates
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(idx, (candidate, &score))| RerankedResult {
                id: candidate.id,
                text: candidate.text.clone(),
                rerank_score: score,
                original_score: candidate.original_score,
                original_rank: candidate.original_rank,
                new_rank: idx, // Will be updated after sorting
                rank_delta: 0, // Will be updated after sorting
            })
            .collect();

        // Sort by rerank score (descending)
        results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply score threshold filter if configured
        if let Some(threshold) = self.config.score_threshold {
            results.retain(|r| r.rerank_score >= threshold);
        }

        // Update new ranks and deltas
        for (new_rank, result) in results.iter_mut().enumerate() {
            result.new_rank = new_rank;
            result.rank_delta = result.original_rank as i32 - new_rank as i32;
        }

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Rerank with batching for large candidate sets
    ///
    /// Processes candidates in batches to manage memory and latency.
    pub async fn rerank_batched(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        top_k: usize,
    ) -> Result<Vec<RerankedResult>> {
        if candidates.len() <= self.config.batch_size {
            return self.rerank(query, candidates, top_k).await;
        }

        let mut all_results = Vec::new();

        // Process in batches
        for chunk in candidates.chunks(self.config.batch_size) {
            let batch_results = self.rerank(query, chunk, chunk.len()).await?;
            all_results.extend(batch_results);
        }

        // Re-sort all results
        all_results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update final ranks
        for (new_rank, result) in all_results.iter_mut().enumerate() {
            result.new_rank = new_rank;
            result.rank_delta = result.original_rank as i32 - new_rank as i32;
        }

        // Take top_k
        all_results.truncate(top_k);

        Ok(all_results)
    }
}

/// Statistics for reranking operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankStats {
    /// Number of candidates reranked
    pub candidates_count: usize,

    /// Average rerank score
    pub avg_rerank_score: f32,

    /// Average rank delta (how much positions shifted)
    pub avg_rank_delta: f32,

    /// Number of results that moved up
    pub moved_up_count: usize,

    /// Number of results that moved down
    pub moved_down_count: usize,

    /// Latency in milliseconds
    pub latency_ms: u64,
}

impl RerankStats {
    /// Compute statistics from reranked results
    pub fn from_results(results: &[RerankedResult], latency_ms: u64) -> Self {
        if results.is_empty() {
            return Self {
                latency_ms,
                ..Default::default()
            };
        }

        let sum_score: f32 = results.iter().map(|r| r.rerank_score).sum();
        let sum_delta: i32 = results.iter().map(|r| r.rank_delta).sum();
        let moved_up = results.iter().filter(|r| r.rank_delta > 0).count();
        let moved_down = results.iter().filter(|r| r.rank_delta < 0).count();

        Self {
            candidates_count: results.len(),
            avg_rerank_score: sum_score / results.len() as f32,
            avg_rank_delta: sum_delta as f32 / results.len() as f32,
            moved_up_count: moved_up,
            moved_down_count: moved_down,
            latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candidates() -> Vec<RerankerCandidate> {
        vec![
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "Machine learning is a subset of artificial intelligence.".to_string(),
                original_score: 0.8,
                original_rank: 0,
            },
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "Deep learning uses neural networks for pattern recognition.".to_string(),
                original_score: 0.7,
                original_rank: 1,
            },
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "The weather today is sunny and warm.".to_string(),
                original_score: 0.6,
                original_rank: 2,
            },
        ]
    }

    #[tokio::test]
    async fn test_reranker_basic() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning AI", &candidates, 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);

        // First result should be the most relevant (machine learning)
        assert!(results[0].text.contains("Machine learning"));

        // Weather result should be last (least relevant to query)
        assert!(results[2].text.contains("weather"));
    }

    #[tokio::test]
    async fn test_reranker_top_k() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning", &candidates, 1)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_reranker_empty() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        let results = reranker.rerank("test query", &[], 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_with_threshold() {
        let mut config = RerankerConfig::default();
        config.score_threshold = Some(0.5);
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning", &candidates, 10)
            .await
            .unwrap();

        // Only results above threshold should be returned
        assert!(results.iter().all(|r| r.rerank_score >= 0.5));
    }

    #[test]
    fn test_heuristic_overlap() {
        let score = HeuristicCrossEncoder::term_overlap_score(
            "machine learning neural",
            "Machine learning is a powerful technique for neural networks.",
        );

        // Should have high overlap
        assert!(score > 0.5);

        let low_score = HeuristicCrossEncoder::term_overlap_score(
            "machine learning",
            "The weather is sunny today.",
        );

        // Should have low overlap
        assert!(low_score < 0.2);
    }

    #[test]
    fn test_rerank_stats() {
        let results = vec![
            RerankedResult {
                id: Uuid::new_v4(),
                text: "Test".to_string(),
                rerank_score: 0.9,
                original_score: 0.8,
                original_rank: 2,
                new_rank: 0,
                rank_delta: 2,
            },
            RerankedResult {
                id: Uuid::new_v4(),
                text: "Test 2".to_string(),
                rerank_score: 0.7,
                original_score: 0.9,
                original_rank: 0,
                new_rank: 1,
                rank_delta: -1,
            },
        ];

        let stats = RerankStats::from_results(&results, 50);

        assert_eq!(stats.candidates_count, 2);
        assert_eq!(stats.moved_up_count, 1);
        assert_eq!(stats.moved_down_count, 1);
        assert_eq!(stats.latency_ms, 50);
    }

    #[test]
    fn test_config_presets() {
        let fast = RerankerConfig::fast();
        assert!(fast.max_length < 512);
        assert!(fast.batch_size > 16);

        let quality = RerankerConfig::quality();
        assert!(quality.use_gpu);
    }
}
